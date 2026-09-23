#include "calibrator.hh"
#include "util.h"
#include <iostream>

Calibrator::Calibrator(unsigned read_core_, unsigned probe_core_, unsigned nr_iterations) :
    read_core(read_core_),
    probe_core(probe_core_),
    shm(new uint8_t[PAGE_SIZE]) {
    read_req.store(false);
    read_ack.store(false);
    req_stop.store(false);
    reader = std::make_unique<std::thread>(do_read, this);

    std::cout << "*** Calibrating the cache latency in " << nr_iterations << " iterations." << std::endl;
    Average llc_hit, l1_hit;
    for (unsigned i = 0; i < nr_iterations; ++i) {
        flush(shm);
        sfence();
        invoke_read();
        llc_hit.sample(reload(shm));
    }

    for (unsigned i = 0; i < nr_iterations; ++i) {
        maccess(shm);
        l1_hit.sample(reload(shm));
    }

    llc_hit_latency = llc_hit();
    l1_hit_latency = l1_hit();
    std::cout << "LLC hit " << llc_hit_latency << " c" << std::endl;
    std::cout << "L1 Cache hit " << l1_hit_latency << " c" << std::endl;
}

void Calibrator::invoke_read() {
    read_req.store(true);
    while(!read_ack)
        ;
    read_ack.store(false);
}

Calibrator::~Calibrator() {
    req_stop.store(true);
    invoke_read();
    reader->join();
    delete shm;
}

void Calibrator::do_read(Calibrator *self) {
    pin_processor(self->read_core);
    bool run = true;
    do {
        while(!self->read_req)
            ;
        self->read_req.store(false);
        if (self->req_stop)
            run = false;
        else {
            maccess(self->shm);
            mfence();
        }
        self->read_ack.store(true);
    } while (run);
}