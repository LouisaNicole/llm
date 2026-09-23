#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include "trace-common.h"

static __inline__ uint64_t rdtsc(void) {
    unsigned int hi, lo;
    __asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

static struct tracer {
    int inited;
    FILE *fp;
    uint64_t *traces;
    size_t trace_pos;
    volatile uint64_t critical_flag;
} my_tracer;

#define MAX_TRACES 1000000

void trace_exit(void) {
    my_tracer.fp = fopen("trace.csv", "w");
    if (!my_tracer.fp) {
        printf("failed to open trace\n");
        return;
    }
    for(size_t i = 0 ;i < my_tracer.trace_pos;++i) {
        fprintf(my_tracer.fp, "%llu\n", my_tracer.traces[i]);
    }
    fclose(my_tracer.fp);
}

void trace_init(void){
    my_tracer.traces = malloc(sizeof(*my_tracer.traces) * MAX_TRACES);
    my_tracer.inited = 1;
}

static void trace_get_rows(void) {
    assert(!my_tracer.critical_flag);
    my_tracer.critical_flag = 1;

    if (!my_tracer.inited)
        trace_init();
    if (my_tracer.trace_pos == MAX_TRACES) {
        printf("tracer: out of buffers, exiting\n");
        exit(1);
    }
    my_tracer.traces[my_tracer.trace_pos++] = rdtsc();

    my_tracer.critical_flag = 0;
}
