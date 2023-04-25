#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

const int SAMPLE_SIZE = 1;

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
    //srand(time(NULL));
    int n, d, s, minv, output;
    //n = (int*)malloc(SAMPLE_SIZE * sizeof(int));

    printf("Give me a number you want to be tested: ");
    scanf("%d", &n);

    if (n % 2 == 0)
    {
        printf("The number needs to be odd.");
        return;
    }
    
    d = n - 1;
    s = 0;

    while(d % 2 == 0)
    {
        s++;
        d = d/2;
    }
    printf("d:%d, s:%d\n", d, s);

    minv = fmin(n - 2, floor(2*(pow(log(n), (2)))));

    // Create the device buffer // dobozolás
    cl_mem n_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
    cl_mem d_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
    cl_mem s_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
    cl_mem minv_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&n);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&d);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&s);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&minv);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&output_buffer);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, NULL);

    // Host buffer -> Device buffer
    cl_int hiba_2 = clEnqueueWriteBuffer(
        command_queue,
        n_buffer,
        CL_FALSE,
        0,
        sizeof(int),
        &n,
        0,
        NULL,
        NULL
    );
    //printf("%d\n", hiba_2);

    clEnqueueWriteBuffer(
        command_queue,
        d_buffer,
        CL_FALSE,
        0,
        sizeof(int),
        &d,
        0,
        NULL,
        NULL
    );

    clEnqueueWriteBuffer(
        command_queue,
        s_buffer,
        CL_FALSE,
        0,
        sizeof(int),
        &s,
        0,
        NULL,
        NULL
    );

    cl_int hiba_3 = clEnqueueWriteBuffer(
        command_queue,
        minv_buffer,
        CL_FALSE,
        0,
        sizeof(int),
        &minv,
        0,
        NULL,
        NULL
    );
    //printf("\n%d\n", hiba_3);

    // Size specification
    size_t local_work_size = 256; // 0-255-ig local ID
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    cl_int hiba = clEnqueueNDRangeKernel(
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
    //printf("\n%d\n",hiba);
    clFinish(command_queue);

    // Host buffer <- Device buffer
    // There's nothing to return from it.
    clEnqueueReadBuffer(
        command_queue,
        output_buffer,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(int),
        &output,
        0,
        NULL,
        NULL
    );

    //printf("output=%d\n",output);
    if( output == 0 )
    {
        printf("Composite.");
    } else if( output == 1 )
    {
        printf("Prime.");
    }
    //printf("\n%d\n",output);

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(n);
    free(d);
    free(s);
    free(output);
    free(minv);
}
