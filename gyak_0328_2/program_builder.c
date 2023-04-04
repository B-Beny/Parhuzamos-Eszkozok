#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


void kernel_loader(const char* const path)
{
    cl_int err;
    int error_code;

    const char* kernel_code = load_kernel_source("kernels/sample.cl", &error_code);
    if (error_code != 0) {
        printf("Source code loading error!\n");
        return 0;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        return 0;
    }
    
    cl_kernel kernel = clCreateKernel(program, "rand_vector", NULL);
}