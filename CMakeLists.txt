cmake_minimum_required(VERSION 3.5)
project(RNN C)

set(CMAKE_C_STANDARD 11)

set (CMAKE_C_FLAGS "-Wall -pedantic -Werror -o4")
add_executable(RNN src/main.c src/jordan-elman.c src/include/jordan-elman.h)
target_link_libraries(RNN -lm -lgsl -lgslcblas)
