#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <cstring>
#include <set>
#include <fstream>

#include <fcntl.h>
#include <unistd.h>
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#include <pthread.h>

#ifdef _WIN32
#include <Windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <sys/wait.h>
#include <spawn.h>
#endif

#include <sstream>
#include <string>
#include <iostream>

#include <time.h>

#include "util.h"

#include <cpuid.h> // Include header for CPUID intrinsics
#include <chrono>
#include <thread>
#include <cmath>
double get_tsc_freq() {
    unsigned int regs[4]; // Array to hold CPUID results

    // Read the 'Time Stamp Counter and Nominal Core Crystal Clock Information Leaf' (leaf 0x15)
    /*
        If EBX[31:0] is 0, the TSC/”core crystal clock” ratio is not enumerated.
        EBX[31:0]/EAX[31:0] indicates the ratio of the TSC frequency and the core crystal clock frequency.
        If ECX is 0, the nominal core crystal clock frequency is not enumerated.
        “TSC frequency” = “core crystal clock frequency” * EBX/EAX.
        The core crystal clock may differ from the reference clock, bus clock, or core clock frequencies.
        EAX Bits 31-00: An unsigned integer which is the denominator of the TSC/”core crystal clock” ratio.
        EBX Bits 31-00: An unsigned integer which is the numerator of the TSC/”core crystal clock” ratio.
        ECX Bits 31-00: An unsigned integer which is the nominal frequency of the core crystal clock in Hz.
        EDX Bits 31-00: Reserved = 0.
    */
    __cpuidex((int*)regs, 0x15, 0);

#if VERBOSE
    // Print out the values from the leaf
    printf("EAX: %08X\n", regs[0]);
    printf("EBX: %08X\n", regs[1]);
    printf("ECX: %08X\n", regs[2]);
    printf("EDX: %08X\n", regs[3]);
#endif

    if (!regs[1]) {
        printf("TSC/'core crystal clock' ratio is not enumerated\n");
        return 1;
    }

    auto crystalClkRatio = double(regs[1])/regs[0];

#if VERBOSE
    printf("TSC/'core crystal clock' ratio = %lf (%u/%u)\n", crystalClkRatio,
            regs[1], regs[0]);
#endif
    if (!regs[2]) {
        printf("nominal core crystal clock frequency is not enumerated\n");
        return 1;
    }

    auto nominalCrystalClk = regs[2]; // Hz
    auto tscFreq = nominalCrystalClk * crystalClkRatio; // Hz

#if VERBOSE
    printf("nominal frequency of the core crystal clock = %lf MHz (%u Hz)\n", double(nominalCrystalClk)/1e6,
            nominalCrystalClk);
    printf("TSC frequency = %lf GHz\n", double(tscFreq)/1e9);
#endif

    uint64_t t0 = rdtsc();
    std::chrono::milliseconds duration(1000);
    std::this_thread::sleep_for(duration);
    uint64_t tsc_1s = rdtsc() - t0;
    double err = tsc_1s * (1/tscFreq) - 1;
    if (fabs(err) > 0.001) {
        printf("verification failed\n");
        assert(0);
    } else {
#if VERBOSE
        printf("OK, difference between systemtime and TSC = %lfsec\n", err);
#endif
    }

    return tscFreq; // Hz
}

void random_shuffle(unsigned *arr, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        unsigned j = rand() % n;
        unsigned temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void pin_processor(unsigned core) {
#ifdef _WIN32
    if (!SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1)<<core)) {
        printf("SetThreadAffinityMask %lu\n", GetLastError());
        exit(1);
    }
#else
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
        printf("pthread_setaffinity_np\n");
        exit(1);
    }
#endif
}

void set_highest_unprivileged_priority() {
#ifdef _WIN32
    if (!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS)) {
        printf("SetPriorityClass %lu\n", GetLastError());
        exit(1);
    }

    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) {
        printf("SetThreadPriority %lu\n", GetLastError());
        exit(1);
    }
#else
    struct sched_param param;
    if (sched_getparam(0, &param) == -1) {
        perror("sched_getparam failed");
        exit(EXIT_FAILURE);
    }

    int policy = SCHED_OTHER; // SCHED_RR
    param.sched_priority = sched_get_priority_max(policy);

    if (sched_setscheduler(0, policy, &param) == -1) {
        perror("sched_setscheduler failed");
        exit(EXIT_FAILURE);
    }
#endif
}

size_t phy_cores() {
#ifdef _WIN32
    DWORD len = 0;
    if (GetLogicalProcessorInformation(nullptr, &len) == FALSE && GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        std::cerr << "GetLogicalProcessorInformation(): Error getting buffer size." << std::endl;
        return 0;
    }
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (GetLogicalProcessorInformation(buffer.data(), &len) == FALSE) {
        std::cerr << "GetLogicalProcessorInformation(): Error getting processor information." << std::endl;
        return 0;
    }

    unsigned int physicalCoreCount = 0;
    size_t count = buffer.size();
    for (size_t i = 0; i < count; ++i) {
        if (buffer[i].Relationship == RelationProcessorCore) {
            // 每个 Relationship 为 RelationProcessorCore 的项代表一个物理核心（可能包含多个逻辑处理器，代表超线程）
            ++physicalCoreCount;
        }
    }
    return physicalCoreCount;
#else
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo) {
        std::cerr << "Cannot open /proc/cpuinfo" << std::endl;
        return 0;
    }

    std::set<std::pair<int, int>> coreSet; // (physical id, core id)
    std::string line;
    int currentPhysicalId = -1;
    int currentCoreId = -1;
    while (std::getline(cpuinfo, line)) {
        if (line.empty()) {
            // 一个 processor 的信息结束时，将记录加入set，然后重置
            if (currentPhysicalId != -1 && currentCoreId != -1) {
                coreSet.insert(std::make_pair(currentPhysicalId, currentCoreId));
            }
            currentPhysicalId = -1;
            currentCoreId = -1;
        } else {
            std::istringstream iss(line);
            std::string key;
            if (std::getline(iss, key, ':')) {
                std::string value;
                if (std::getline(iss, value)) {
                    // strip spaces 
                    auto trim = [](std::string& s) {
                        size_t first = s.find_first_not_of(" \t");
                        size_t last = s.find_last_not_of(" \t");
                        if (first != std::string::npos && last != std::string::npos)
                            s = s.substr(first, last - first + 1);
                        else
                            s = "";
                    };
                    trim(key);
                    trim(value);
                    if (key == "physical id") {
                        currentPhysicalId = std::stoi(value);
                    } else if (key == "core id") {
                        currentCoreId = std::stoi(value);
                    }
                }
            }
        }
    }
    // 处理最后一个块（如果文件没有以空行结束）
    if (currentPhysicalId != -1 && currentCoreId != -1) {
        coreSet.insert(std::make_pair(currentPhysicalId, currentCoreId));
    }

    return coreSet.size();
#endif
}

/* https://www.cnblogs.com/sigma0/p/10837760.html */
void* aligned_malloc(size_t required_bytes, size_t alignment) {
    int offset = alignment - 1 + sizeof(void*) /* org ptr */;
    void* p1 = (void*)malloc(required_bytes + offset);
    if (!p1)
        return nullptr;
    void** p2 = (void**)( ( (uintptr_t)p1 + offset) & ~(alignment - 1) );
    p2[-1] = p1;
    return p2;
}
void aligned_free(void *p2) {
    free(((void**)p2)[-1]);
}

void set_page_executable(void *mem, size_t size, unsigned modify) {
#ifdef _WIN32
    DWORD oldProtect;
    if (!VirtualProtect(mem, size, modify ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ, &oldProtect)) {
        printf("Failed to set memory protection. Error code: %lu\n", GetLastError());
        exit(1);
    }
#else
    if (mprotect(mem, size, PROT_READ | (modify ? PROT_WRITE : 0) | PROT_EXEC)) {
        perror("mprotect");
        exit(1);
    }
#endif
}


void *get_shared_memory(const char *file, unsigned long *size) {
#ifdef _WIN32
    HANDLE dumpFileDescriptor = CreateFileA(file,
                        GENERIC_READ,
                        FILE_SHARE_READ /*| FILE_SHARE_WRITE*/,
                        NULL,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        NULL);
    if (dumpFileDescriptor == INVALID_HANDLE_VALUE) {
        printf("CreateFileA %lu\n", GetLastError());
        exit(1);
    }
    HANDLE fileMappingObject = CreateFileMapping(dumpFileDescriptor,
                        NULL,
                        PAGE_READONLY,
                        0,
                        0,
                        NULL);
    if (fileMappingObject == INVALID_HANDLE_VALUE) {
        printf("CreateFileMapping %lu\n", GetLastError());
        exit(1);
    }
    void* mappedFileAddress = MapViewOfFile(fileMappingObject,
                        FILE_MAP_READ,
                        0,
                        0,
                        0/*MMAP_ALLOCATOR_SIZE*/);
    if (!mappedFileAddress) {
        printf("MapViewOfFile %lu\n", GetLastError());
        exit(1);
    }
    return mappedFileAddress;
#else
    int fd = open(file, O_RDONLY);
    if (fd < 0 ) {
        std::cerr << "Error opening file:" << file << std::endl;
        exit(EXIT_FAILURE);
    }

    struct stat st_buf;
    fstat(fd, &st_buf);
    *size = st_buf.st_size;

    void *rv = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (rv == MAP_FAILED) {
        perror("Error mapping file.");
        exit(EXIT_FAILURE);
    }
    return rv;
#endif
}

#ifdef _WIN32
void *map_exec(const char *binfile) {
    HMODULE hmodule = LoadLibraryA(binfile);
    if (!hmodule) {
        printf("LoadLibraryA %lu\n", GetLastError());
        exit(1);
    }
    MODULEINFO modinfo;
    BOOL ret = GetModuleInformation(GetCurrentProcess(), hmodule, &modinfo, sizeof(modinfo));
    if (!ret) {
        printf("GetModuleInformation %lu\n", GetLastError());
        exit(1);
    }
    return modinfo.lpBaseOfDll;
}
#endif

#ifdef _WIN32
void *mapview(const char *file, unsigned long *size,
                size_t view_offset, size_t view_size) {
    assert(view_offset % PAGE_SIZE == 0 && view_size % PAGE_SIZE == 0);
    HANDLE dumpFileDescriptor = CreateFileA(file,
                        GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE,
                        NULL,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        NULL);
    if (dumpFileDescriptor == INVALID_HANDLE_VALUE) {
        printf("CreateFileA %lu\n", GetLastError());
        exit(1);
    }
    *size = GetFileSize(dumpFileDescriptor, nullptr);
    if (*size == INVALID_FILE_SIZE) {
        printf("GetFileSize %lu\n", GetLastError());
        exit(1);
    }
    HANDLE fileMappingObject = CreateFileMapping(dumpFileDescriptor,
                        NULL,
                        PAGE_READONLY,
                        0,
                        0,
                        NULL);
    if (fileMappingObject == INVALID_HANDLE_VALUE) {
        printf("CreateFileMapping %lu\n", GetLastError());
        exit(1);
    }
    assert(view_size && view_offset + view_size < *size);
    void* mappedFileAddress = MapViewOfFile(fileMappingObject,
                        FILE_MAP_READ,
                        DWORD(view_offset & 0xffffffff),
                        DWORD(view_offset >> 32),
                        view_size/*MMAP_ALLOCATOR_SIZE*/);
    if (!mappedFileAddress) {
        printf("MapViewOfFile %lu\n", GetLastError());
        exit(1);
    }
    return mappedFileAddress;
}
#endif

size_t get_page_size() {
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize;
#else
    return getpagesize();
#endif
}

size_t get_phy_ram_size() {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    return statex.ullTotalPhys;
#else
    struct sysinfo info;
    sysinfo(&info);
    return (size_t) info.totalram * (size_t) info.mem_unit;
#endif
}

void yield_cpu() {
#ifdef _WIN32
    Sleep(0);
#else
    sched_yield();
#endif
}

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
void llama_mlock::init(void * ptr) {
    assert(addr == NULL && size == 0); // NOLINT
    addr = ptr;
}

llama_mlock::~llama_mlock() {
    if (size) {
        raw_unlock(addr, size);
    }
}

#ifdef _WIN32
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}

static size_t lock_granularity() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
}

bool llama_mlock::raw_lock(void * ptr, size_t len) const {
    for (int tries = 1; ; tries++) {
        if (VirtualLock(ptr, len)) {
            return true;
        }
        if (tries == 2) {
            fprintf(stderr, "warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                len, size, llama_format_win_err(GetLastError()).c_str());
            return false;
        }

        // It failed but this was only the first try; increase the working
        // set size and try again.
        SIZE_T min_ws_size, max_ws_size;
        if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
            fprintf(stderr, "warning: GetProcessWorkingSetSize failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
            return false;
        }
        // Per MSDN: "The maximum number of pages that a process can lock
        // is equal to the number of pages in its minimum working set minus
        // a small overhead."
        // Hopefully a megabyte is enough overhead:
        size_t increment = len + 1048576;
        // The minimum must be <= the maximum, so we need to increase both:
        min_ws_size += increment;
        max_ws_size += increment;
        if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
            fprintf(stderr, "warning: SetProcessWorkingSetSize failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
            return false;
        }
    }
}

void llama_mlock::raw_unlock(void * ptr, size_t len) {
    if (!VirtualUnlock(ptr, len)) {
        fprintf(stderr, "warning: failed to VirtualUnlock buffer: %s\n",
                llama_format_win_err(GetLastError()).c_str());
    }
}
#else
static size_t lock_granularity() {
    return (size_t) sysconf(_SC_PAGESIZE);
}

#ifdef __APPLE__
    #define MLOCK_SUGGESTION \
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
        "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
    #define MLOCK_SUGGESTION \
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

bool llama_mlock::raw_lock(void * addr, size_t size) const {
    if (!mlock(addr, size)) {
        return true;
    }

    char* errmsg = std::strerror(errno);
    bool suggest = (errno == ENOMEM);

    // Check if the resource limit is fine after all
    struct rlimit lock_limit;
    if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
        suggest = false;
    }
    if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
        suggest = false;
    }

    fprintf(stderr, "warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
            size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
    return false;
}

#undef MLOCK_SUGGESTION

void llama_mlock::raw_unlock(void * addr, size_t size) {
    if (munlock(addr, size)) {
        fprintf(stderr, "warning: failed to munlock buffer: %s\n", std::strerror(errno));
    }
}
#endif

bool llama_mlock::grow_to(size_t target_size) {
    assert(addr);
    if (failed_already) {
        return false;
    }
    size_t granularity = lock_granularity();
    target_size = (target_size + granularity - 1) & ~(granularity - 1);
    if (target_size > size) {
        if (raw_lock((uint8_t *) addr + size, target_size - size)) {
            size = target_size;
        } else {
            failed_already = true;
            return false;
        }
    }
    return true;
}

struct Process {
#ifdef _WIN32
    PROCESS_INFORMATION pi;
#else
    pid_t pid;
#endif
};

Process *create_process(const char *argv[], unsigned argc, bool disable_io) {
    auto proc = new Process();
#ifdef _WIN32
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    if (disable_io) {
        si.hStdInput = INVALID_HANDLE_VALUE;
        si.hStdOutput = INVALID_HANDLE_VALUE;
        si.hStdError = INVALID_HANDLE_VALUE;
        si.dwFlags |= STARTF_USESTDHANDLES;
    }
    ZeroMemory(&proc->pi, sizeof(proc->pi));

    std::stringstream ss;
    ss << argv[0];
    for (unsigned i=1; i<argc; ++i) {
        ss << " " << argv[i];
    }

    char *cmdline = strdup(ss.str().c_str());
    if (!CreateProcessA(NULL, 
        cmdline, 
        NULL, 
        NULL, 
        FALSE, 
        0, 
        NULL, 
        NULL, 
        &si, 
        &proc->pi)) {
        free(cmdline);
        std::cerr << "CreateProcess failed (" << GetLastError() << ")." << std::endl;
        return nullptr;
    }
    free(cmdline);
#else
    char **argv_ = new char *[argc+1];
    for (unsigned i=0; i<argc; ++i) {
        argv_[i] = strdup(argv[i]);
    }
    argv_[argc] = nullptr;

    extern char **environ;
    
    int err;
    posix_spawn_file_actions_t action;
    posix_spawnattr_t attr;
    err = posix_spawnattr_init(&attr);
    if (err) {
        fprintf(stderr, "posix_spawnattr_init() failed: %d\n", err);
        return nullptr;
    }
    err = posix_spawn_file_actions_init(&action);
    if (err) {
        fprintf(stderr, "posix_spawn_file_actions_init() failed: %d\n", err);
        return nullptr;
    }
    if (disable_io) {
        err = posix_spawn_file_actions_addopen(&action, STDIN_FILENO, "/dev/null", O_WRONLY, 0);
        if (err) {
            fprintf(stderr, "posix_spawn_file_actions_addopen() failed for STDOUT: %d\n", err);
            return nullptr;
        }
        err = posix_spawn_file_actions_addopen(&action, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
        if (err) {
            fprintf(stderr, "posix_spawn_file_actions_addopen() failed for STDOUT: %d\n", err);
            return nullptr;
        }
        err = posix_spawn_file_actions_addopen(&action, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
        if (err) {
            fprintf(stderr, "posix_spawn_file_actions_addopen() failed for STDERR: %d\n", err);
            return nullptr;
        }
    }

    if( (err=posix_spawn(&proc->pid, argv_[0], &action, &attr, argv_, environ)) !=0 ) {
        for (unsigned i=0; i<argc; ++i) {
            delete argv_[i];
        }
        delete []argv_;
        
        perror("posix_spawn()");
        return nullptr;
    }

    for (unsigned i=0; i<argc; ++i) {
        delete argv_[i];
    }
    delete []argv_;
#endif
    return proc;
}

/// @return true if the target has been terminated
bool wait_process(Process *proc, int &status) {
#ifdef _WIN32
    DWORD ret = WaitForSingleObject(proc->pi.hProcess, 0);
    if (WAIT_OBJECT_0 == ret) {
        DWORD exit_code;
        GetExitCodeProcess(proc->pi.hProcess, &exit_code);
        status = exit_code;
        return true;
    } else if (WAIT_TIMEOUT != ret) {
        std::cerr << "WaitForSingleObject()" << std::endl;
    }
    return false;
#else
    int ret;
    ret = waitpid(proc->pid, &status, WNOHANG);
    if (ret < 0)
        perror("waitpid()");
    return !!ret;
#endif
}

/// @return true if succeeded
bool wait_process_blocked(Process *proc, int &status) {
#ifdef _WIN32
    DWORD ret = WaitForSingleObject(proc->pi.hProcess, INFINITE);
    if (WAIT_OBJECT_0 == ret) {
        DWORD exit_code;
        GetExitCodeProcess(proc->pi.hProcess, &exit_code);
        status = exit_code;
    } else if (WAIT_TIMEOUT != ret) {
        std::cerr << "WaitForSingleObject()" << std::endl;
        return false;
    }
#else
    int ret;
    ret = waitpid(proc->pid, &status, 0);
    if (ret < 0) {
        perror("waitpid()");
        return false;
    }
#endif
    return true;
}

void free_process(Process *&proc) {
#ifdef _WIN32
    CloseHandle(proc->pi.hProcess);
    CloseHandle(proc->pi.hThread);
#endif
    delete proc;
    proc = nullptr;
}

#ifdef _WIN32
struct NamedPipe {
    std::string filename;
    HANDLE h_file;
};

NamedPipe *create_named_pipe() {
    auto ret = new NamedPipe();
    std::stringstream pipe_name;
    pipe_name << "\\\\.\\pipe\\ikwys-" << time(nullptr);

    ret->filename = pipe_name.str();
    ret->h_file = CreateNamedPipeA(ret->filename.c_str(), 
        PIPE_ACCESS_DUPLEX|FILE_FLAG_WRITE_THROUGH,
        PIPE_TYPE_BYTE|PIPE_READMODE_BYTE|PIPE_WAIT,
        PIPE_UNLIMITED_INSTANCES, 0, 0, 0, NULL);
    if (INVALID_HANDLE_VALUE == ret->h_file) {
        std::cerr << "CreateNamedPipeA() err " << GetLastError() << std::endl;
        delete ret;
        return nullptr;
    }

    return ret;
}

char *get_named_pipe_filename(NamedPipe *pipe) {
    return strdup(pipe->filename.c_str());
}

void destory_named_pipe(NamedPipe *&pipe) {
    CloseHandle(pipe->h_file);
    delete pipe;
    pipe = nullptr;
}
#endif

const char *get_temp_dir() {
#ifdef _WIN32
    return ".";
#else
    return "/tmp";
#endif
}

static bool tsc2sec_inited;
static double tsc_freq;
double tsc2sec(uint64_t tsc) {
    if (!tsc2sec_inited) {
        tsc_freq = get_tsc_freq();
        tsc2sec_inited = true;
    }
    return tsc/tsc_freq;
}
