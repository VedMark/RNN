cmake_minimum_required(VERSION 3.9)
project(RNN C)

set(CMAKE_C_STANDARD 11)

add_executable(RNN src/main.c src/jordan-elman.c src/include/jordan-elman.h src/algebra.c src/include/algebra.h)
target_link_libraries(RNN -lm )