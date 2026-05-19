#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "tga.h"

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        printf("Error during %s: %d\n", operation, err);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    cl_int err;

    // --- Platform ---
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "clGetPlatformIDs (count)");
    printf("Found %u OpenCL platform(s)\n", numPlatforms);

    if (numPlatforms == 0) {
        printf("No OpenCL platforms found!\n");
        return EXIT_FAILURE;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    checkError(err, "clGetPlatformIDs");

    cl_platform_id platform = platforms[0];
    char platformName[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
    printf("Using platform: %s\n", platformName);

    // --- Device (prefer GPU, fallback to CPU) ---
    cl_device_id device;
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        printf("No GPU found, falling back to CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &numDevices);
        checkError(err, "clGetDeviceIDs CPU");
    }

    // --- Device Info ---
    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("Using device: %s\n", deviceName);

    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    printf("Max work group size: %zu\n", maxWorkGroupSize);

    cl_uint maxWorkItemDims;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDims), &maxWorkItemDims, NULL);

    size_t maxWorkItemSizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, NULL);
    printf("Max work item sizes:");
    for (cl_uint i = 0; i < maxWorkItemDims; i++) {
        printf(" %zu", maxWorkItemSizes[i]);
    }
    printf("\n");

    cl_ulong globalMemSize;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
    printf("Global memory: %llu MB\n", (unsigned long long)(globalMemSize / (1024 * 1024)));

    // --- Context & Command Queue ---
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    printf("\nOpenCL setup complete.\n");

    // cleanup
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(platforms);

    return EXIT_SUCCESS;
}
