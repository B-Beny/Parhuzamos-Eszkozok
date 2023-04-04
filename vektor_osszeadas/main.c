#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

typedef struct vectors
{
    int* v1;
    int* v2;
    int* output;
} Vectors;


const int SAMPLE_SIZE = 5;

int main(void)
{
    int i;
    cl_int err;
    int error_code;

    // Get platform
    cl_uint n_platforms;
	cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
		return 0;
	}

    // Get device
	cl_device_id device_id;
	cl_uint n_devices;
	err = clGetDeviceIDs(
		platform_id,
		CL_DEVICE_TYPE_GPU,
		1,
		&device_id,
		&n_devices
	);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
		return 0;
	}

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
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
    cl_kernel kernel = clCreateKernel(program, "sample_kernel", NULL);

    // Create the host buffer and initialize it
    Vectors v;
    v.v1 = (int*)malloc(SAMPLE_SIZE * sizeof(int));
    v.v2 = (int*)malloc(SAMPLE_SIZE * sizeof(int));
    v.output = (int*)malloc(SAMPLE_SIZE * sizeof(int));

    for (int i = 0; i < SAMPLE_SIZE; i++)
    {
        v.v1[i] = i;
        v.v2[i] = i + 2;
    }
    

    // Create the device buffer // dobozolás
    cl_mem v1_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
    cl_mem v2_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&v1_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&v2_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&SAMPLE_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        v1_buffer,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(int),
        v.v1,
        0,
        NULL,
        NULL
    );
    clEnqueueWriteBuffer(
        command_queue,
        v2_buffer,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(int),
        v.v2,
        0,
        NULL,
        NULL
    );
    

    // Size specification
    size_t local_work_size = 256; // 0-255-ig local ID
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );
    clFinish(command_queue);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        output_buffer,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(int),
        v.output,
        0,
        NULL,
        NULL
    );

    for (i = 0; i < SAMPLE_SIZE; ++i) {
        printf("[%d] = %d\n", i, v.output[i]);
    }

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(v.v1);
    free(v.v2);
    free(v.output);
}
