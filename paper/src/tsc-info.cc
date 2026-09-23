
#include <stdio.h>
#include "util.h"

int main() {
    
    printf("TSC frequency = %lf GHz\n", double(get_tsc_freq())/1e9);

    return 0;
}