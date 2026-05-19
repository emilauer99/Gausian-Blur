#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "tga.h"

#define FILTER_SIZE 5
#define SIGMA 1.0

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        printf("Error during %s: %d\n", operation, err);
        exit(EXIT_FAILURE);
    }
}

// generate normalized 2D Gaussian filter kernel
void generateGaussianKernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int half = size / 2;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float x = (float)(i - half);
            float y = (float)(j - half);
            kernel[i * size + j] = (1.0f / (2.0f * (float)M_PI * sigma * sigma))
                * expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            sum += kernel[i * size + j];
        }
    }

    // normalize to preserve brightness
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

int main(int argc, char* argv[]) {
    cl_int err;

    // --- Load Input Image ---
    const char* inputFile = "shuttle.tga";
    const char* outputFile = "output.tga";

    tga::TGAImage image;
    if (!tga::LoadTGA(&image, inputFile)) {
        printf("Failed to load image: %s\n", inputFile);
        return EXIT_FAILURE;
    }
    printf("Loaded image: %s (%ux%u, %u bpp)\n", inputFile, image.width, image.height, image.bpp);

    unsigned int bytesPerPixel = image.bpp / 8;
    unsigned int imageSize = image.width * image.height * bytesPerPixel;

    // --- Generate Gaussian Filter Kernel ---
    int filterSize = FILTER_SIZE;
    float filterKernel[FILTER_SIZE * FILTER_SIZE];
    generateGaussianKernel(filterKernel, filterSize, SIGMA);

    printf("Gaussian kernel (%dx%d, sigma=%.1f):\n", filterSize, filterSize, SIGMA);
    for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
            printf("%.6f ", filterKernel[i * filterSize + j]);
        }
        printf("\n");
    }

    // --- Platform ---
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "clGetPlatformIDs (count)");

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
    printf("\nUsing platform: %s\n", platformName);

    // --- Device (prefer GPU, fallback to CPU) ---
    cl_device_id device;
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        printf("No GPU found, falling back to CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &numDevices);
        checkError(err, "clGetDeviceIDs CPU");
    }

    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("Using device: %s\n", deviceName);

    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    printf("Max work group size: %zu\n", maxWorkGroupSize);

    size_t maxWorkItemSizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, NULL);
    printf("Max work item sizes: %zu %zu %zu\n", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

    // validate NDRange against device capabilities
    if (image.width > maxWorkItemSizes[0] || image.height > maxWorkItemSizes[1]) {
        printf("Error: image dimensions (%ux%u) exceed max work item sizes (%zux%zu)\n",
            image.width, image.height, maxWorkItemSizes[0], maxWorkItemSizes[1]);
        free(platforms);
        return EXIT_FAILURE;
    }

    // --- Context & Command Queue ---
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    // --- Create Buffers ---
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        imageSize, image.imageData.data(), &err);
    checkError(err, "clCreateBuffer (input)");

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        imageSize, NULL, &err);
    checkError(err, "clCreateBuffer (output)");

    cl_mem filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * filterSize * filterSize, filterKernel, &err);
    checkError(err, "clCreateBuffer (filter)");

    printf("\nBuffers created (input: %u bytes, filter: %dx%d)\n", imageSize, filterSize, filterSize);

    // TODO: Sprint 3 - load kernel, execute, read back result

    // cleanup
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(inputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(platforms);

    return EXIT_SUCCESS;
}
