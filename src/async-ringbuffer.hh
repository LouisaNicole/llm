#ifndef ASYNC_QUEUE_HH_
#define ASYNC_QUEUE_HH_

#include <cassert>
#include <cstdint>
#include <pthread.h>
#include <semaphore.h>

#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>

template<typename T>
class AsyncRingBuffer {
    std::mutex pool_lock, chain_lock;
    std::atomic<bool> stopped;
    
public:
    struct node : public std::vector<T> {
    };
private:
    uint8_t *mem_pool;
    std::queue<node *> pool;
    std::queue<node *> que;
    sem_t que_full;

public:
    AsyncRingBuffer(size_t reserved) {
        sem_init(&que_full, 0, 0);
        // allocate a continuous large chunk
        mem_pool = new uint8_t[sizeof(node) * reserved];
        for (size_t i = 0; i < reserved; ++i) {
            pool.push((node *)::operator new(sizeof(node), mem_pool));
            mem_pool += sizeof(node);
        }
    }
    ~AsyncRingBuffer() {
        // TODO free mem_pool
    }

    node *alloc_slow() {
        node *cur;
        std::unique_lock<std::mutex> _lck(pool_lock);
        if (pool.empty()) {
#if 0 // slow
            cur = new node();
            pool.emplace(cur);
#else
            std::cout << "pool oom" << std::endl;
            abort();
#endif
        } else {
            cur = pool.front();
            pool.pop();
        }
        return cur;
    }
    void add_slow(node *cur) {
        if (cur->empty()) {
            free_slow(cur);
        } else {
            {
                std::unique_lock<std::mutex> _lck(chain_lock);
                que.emplace(cur);
            }
            int ret = sem_post(&que_full);
            assert(!ret);
        }
    }
    
    node *get_slow() {
        for(;;) {
            int status = sem_wait(&que_full);
            if (status) {
                assert(errno == EINTR);
                continue; /* retry if we are signaled */
            } else {
                break;
            }
        }

        std::unique_lock<std::mutex> _lck(chain_lock);
        if (que.empty())
            return nullptr; // stopped
        node *ret = ret = que.front();
        que.pop();
        return ret;
    }
    node *get_slow_timed(long nsec) {
        timespec ts = {
            .tv_sec = 0,
            .tv_nsec = nsec,
        };
        for(;;) {
            int status = sem_timedwait(&que_full, &ts);
            if (status) {
                if (errno == ETIMEDOUT) {
                    return nullptr;
                } else {
                    assert(errno == EINTR);
                    continue; /* retry if we are signaled */
                }
            } else {
                break;
            }
        }

        std::unique_lock<std::mutex> _lck(chain_lock);
        node *ret = que.front();
        que.pop();
        return ret;
    }

    void put_slow(node *cur) {
        cur->clear();
        free_slow(cur);
    }

private:
    void free_slow(node *cur) {
        std::unique_lock<std::mutex> _lck(pool_lock);
        pool.emplace(cur);
    }
    
};

#endif