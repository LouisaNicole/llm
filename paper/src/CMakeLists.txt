cmake_minimum_required(VERSION 3.0)
project(attk)

add_compile_options(-Wall -Wno-unused-function)
add_compile_options(-Wl,--stack,1073741824)
add_compile_options(-O3)
add_compile_options(-g)

#find_package(Boost 1.36.0 COMPONENTS iostreams)
#include_directories(${Boost_INCLUDE_DIRS})

include_directories(prefetcher/)

add_executable(main
    attack.cc gguf.c tokenizer.cc util.cc main.cc
    rand-constraint-solver.cc
#    prefetcher/base.cc
#    prefetcher/queued.cc
#    prefetcher/stride.cc
    )

#target_link_libraries(main ${Boost_LIBRARIES} elf)

add_executable(exp1
    exp1.cc gguf.c tokenizer.cc util.cc 
)