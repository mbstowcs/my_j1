cmake_minimum_required(VERSION 3.31)
project(my_j1 C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
        "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(my_j1 SHARED library.c)

target_include_directories(my_j1 PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${NUMPY_INCLUDE}
        )

target_link_libraries(my_j1 PRIVATE
        ${Python3_LIBRARIES}
        -static-libgcc
        -static-libstdc++
        -Wl,-Bstatic
        -lgomp
        -lwinpthread
        -Wl,-Bdynamic
        )

target_compile_options(my_j1 PRIVATE
        -fopenmp
        )

set_target_properties(my_j1 PROPERTIES
        PREFIX ""
        SUFFIX ".pyd"
        )
