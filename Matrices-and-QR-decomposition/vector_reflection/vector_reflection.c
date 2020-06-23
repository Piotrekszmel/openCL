#define _CRT_SECURE_NO_WARNINGS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define PROGRAM_FILE "vector_reflection.cl"
#define KERNEL_FUNC "vector_reflection"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif


/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

   /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) 
    {
      perror("Couldn't identify a platform");
      exit(1);
    } 

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    if(err < 0) 
    {
      perror("Couldn't access any devices");
      exit(1);   
    }

    return device;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context context, cl_device_id device, const char* filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err; 

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "rb");
    if(program_handle == NULL) 
    {
      perror("Couldn't find the program file");
      exit(1);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(context, 1, 
        (const char**)&program_buffer, &program_size, &err);
    if(err < 0) 
    {
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
            perror("Couldn't get program build info");
            exit(1);
        }
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
        printf("LOG: %s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}



int main(void)
{
    /* Host/device data structures */
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue queue;
    cl_int err;

    /* Data and buffers */
    float reflect[4];
    cl_mem reflect_buffer;
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float u[4] = {0.0f, 5.0f, 0.0f, 0.0f};
    
    /* Create a device and context */
    device = create_device();
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err < 0) 
    {
      perror("Couldn't create a context");
      exit(1);   
    }
    
    /* Build the program */
    program = build_program(context, device, PROGRAM_FILE);

    /* Create a kernel */
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if(err < 0) 
    {
      perror("Couldn't create a kernel");
      exit(1);
    }
    
    /* Create buffer */
    reflect_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        4 * sizeof(float), NULL, &err);
    if(err < 0) 
    {
      perror("Couldn't create a buffer");
      exit(1);   
    }

    /* Create kernel argument */
    err = clSetKernelArg(kernel, 0, sizeof(x), x);
    err |= clSetKernelArg(kernel, 1, sizeof(u), u);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &reflect_buffer);
    if(err < 0) 
    {
      printf("Couldn't set a kernel argument");
      exit(1);   
    }
    
    /* Create a command queue */
    queue = clCreateCommandQueue(context, device, 0, &err);
    if(err < 0) 
    {
      perror("Couldn't create a command queue");
      exit(1);   
    }

    /* Enqueue kernel */
    err = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    if(err < 0) 
    {
      perror("Couldn't enqueue the kernel");
      exit(1);   
    }

    /* Read and print the result */
    err = clEnqueueReadBuffer(queue, reflect_buffer, CL_TRUE,
        0, sizeof(reflect), reflect, 0, NULL, NULL);
    if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);   
    }
    printf("Result: %f %f %f %f\n", 
         reflect[0], reflect[1], reflect[2], reflect[3]);

    /* Deallocate resources */
    clReleaseMemObject(reflect_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseContext(context);
    clReleaseProgram(program);
    return 0;
}