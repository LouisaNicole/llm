#ifndef CALIBRATOR_HH_
#define CALIBRATOR_HH_

#include <cstdint>
#include <atomic>
#include <thread>
#include <memory>

class Calibrator {
    const unsigned read_core, probe_core;
    uint64_t llc_hit_latency, l1_hit_latency;
    std::atomic<bool> read_req, read_ack, req_stop;
    uint8_t *shm;
    std::unique_ptr<std::thread> reader;
    
    static void do_read(Calibrator *self);
    void invoke_read();
public:
    Calibrator(unsigned read_core, unsigned probe_core, unsigned nr_iterations=10000);
    ~Calibrator();

    inline uint64_t get_llc_hit_latency() const {
        return llc_hit_latency;
    }
    inline uint64_t get_l1_hit_latency() const {
        return l1_hit_latency;
    }
};

#endif // CALIBRATOR_HH_