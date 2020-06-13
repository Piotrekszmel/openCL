#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "string_search.cl"
#define KERNEL_FUNC "string_search"
#define TEXT_FILE "kafka.txt"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else  
#include <CL/cl.h>
#endif

int main()
{
    /* Host/device data structures */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;

    /* Program/kernel data structures */
    cl_program program;
    cl_kernel kernel;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    size_t offset = 0;
    size_t global_size, local_size;

    /* Data and buffers */
    char pattern[16] = "thatwithhavefrom";
    FILE *text_handle;
    char *text;
    size_t text_size;
    int chars_per_item;
    int result[4] = { 0, 0, 0, 0 };
    cl_mem text_buffer, result_buffer;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    }

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err < 0) {
        perror("Couldn't access any GPU devices");
        exit(1);
    }

    /* Determine global size and local size */
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(global_size), &global_size, NULL);
    err |= clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(local_size), &local_size, NULL);
    if (err < 0)
    {
        perror("Could not obtain device information");
        exit(1);
    }
    global_size *= local_size;

    /* Create a context */
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    program_handle = fopen(PROGRAM_FILE, "rb");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)calloc(program_size + 1, sizeof(char));
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    text_handle = fopen(TEXT_FILE, "rb");
    if (text_handle == NULL) {
        perror("Couldn't find the text file");
        exit(1);
    }
    fseek(text_handle, 0, SEEK_END);
    text_size = ftell(text_handle) - 1;
    rewind(text_handle);
    text = (char*)calloc(text_size + 1, sizeof(char));
    text[text_size] = '\0';
    fread(text, sizeof(char), text_size, text_handle);
    fclose(text_handle);
    chars_per_item = text_size / global_size + 1;

    program = clCreateProgramWithSource(context, 1,
        (const char**)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program */
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err < 0)
    {
        /* Find size of log and print to std output */
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        if (err < 0)
        {
            perror("Could not get program build info");
            exit(1);
        }
        program_log = (char*)calloc(log_size + 1, sizeof(char));
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
        printf("LOG: %s\n", program_log);
        free(program_log);
        exit(1);
    }

    /* Create a kernel */
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        exit(1);
    }

    /* Create buffers to hold the text characters and count */
    text_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
        CL_MEM_COPY_HOST_PTR, text_size + 1, text, &err);
    if (err < 0) {
        perror("Couldn't create a test buffer");
        exit(1);
    }

    result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, sizeof(result), result, &err);
    if (err < 0) {
        perror("Couldn't create a result buffer");
        exit(1);
    }

    /* Create kernel argument */
    err = clSetKernelArg(kernel, 0, sizeof(pattern), pattern);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &text_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(chars_per_item), &chars_per_item);
    err |= clSetKernelArg(kernel, 3, 4 * sizeof(int), NULL);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &result_buffer);
    if (err < 0) {
        printf("Couldn't set a kernel argument");
        exit(1);
    }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, &offset, &global_size,
        &local_size, 0, NULL, NULL);
    if (err < 0) {
        perror("Couldn't enqueue the kernel");
        printf("Error code: %d\n", err);
        exit(1);
    }

    err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
        sizeof(result), result, 0, NULL, NULL);
    if (err < 0) {
        perror("Couldn't read the buffer");
        exit(1);
    }

    printf("\nResults: \n");
    printf("Number of occurrences of 'that': %d\n", result[0]);
    printf("Number of occurrences of 'with': %d\n", result[1]);
    printf("Number of occurrences of 'have': %d\n", result[2]);
    printf("Number of occurrences of 'from': %d\n", result[3]);

    clReleaseMemObject(result_buffer);
    clReleaseMemObject(text_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}