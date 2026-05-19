#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "tga.h"

#define DEFAULT_FILTER_SIZE 5
#define DEFAULT_SIGMA 1.0f
#define MAX_FILTER_SIZE 9

// save RGB pixel data as BMP file
bool saveBMP(const char* filename, const unsigned char* data, int width, int height, int bytesPerPixel) {
    int rowPadding = (4 - (width * 3) % 4) % 4;
    int dataSize = (width * 3 + rowPadding) * height;
    int fileSize = 54 + dataSize;

    unsigned char header[54] = {0};
    // BMP header
    header[0] = 'B'; header[1] = 'M';
    header[2] = fileSize; header[3] = fileSize >> 8; header[4] = fileSize >> 16; header[5] = fileSize >> 24;
    header[10] = 54; // pixel data offset
    // DIB header
    header[14] = 40; // DIB header size
    header[18] = width; header[19] = width >> 8; header[20] = width >> 16; header[21] = width >> 24;
    header[22] = height; header[23] = height >> 8; header[24] = height >> 16; header[25] = height >> 24;
    header[26] = 1;  // color planes
    header[28] = 24; // bits per pixel
    header[34] = dataSize; header[35] = dataSize >> 8; header[36] = dataSize >> 16; header[37] = dataSize >> 24;

    FILE* fp = fopen(filename, "wb");
    if (!fp) return false;
    fwrite(header, 1, 54, fp);

    unsigned char padding[3] = {0, 0, 0};
    // BMP stores rows bottom-to-top, TGA data is already bottom-to-top
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * bytesPerPixel;
            unsigned char bgr[3] = { data[idx + 2], data[idx + 1], data[idx] };
            fwrite(bgr, 1, 3, fp);
        }
        if (rowPadding > 0) fwrite(padding, 1, rowPadding, fp);
    }

    fclose(fp);
    return true;
}

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

    // --- Parse Arguments ---
    const char* inputFile = "shuttle.tga";
    const char* outputFile = "output.bmp";
    int filterSize = DEFAULT_FILTER_SIZE;
    float sigma = DEFAULT_SIGMA;

    if (argc >= 2) inputFile = argv[1];
    if (argc >= 3) filterSize = atoi(argv[2]);
    if (argc >= 4) sigma = (float)atof(argv[3]);

    if (filterSize < 1 || filterSize > MAX_FILTER_SIZE || filterSize % 2 == 0) {
        printf("Error: filter size must be odd and between 1 and %d\n", MAX_FILTER_SIZE);
        return EXIT_FAILURE;
    }

    printf("Usage: gaussian_blur [input.tga] [filterSize] [sigma]\n\n");

    // --- Load Input Image ---
    tga::TGAImage image;
    if (!tga::LoadTGA(&image, inputFile)) {
        printf("Failed to load image: %s\n", inputFile);
        return EXIT_FAILURE;
    }
    printf("Loaded image: %s (%ux%u, %u bpp)\n", inputFile, image.width, image.height, image.bpp);

    unsigned int bytesPerPixel = image.bpp / 8;
    unsigned int imageSize = image.width * image.height * bytesPerPixel;

    // --- Generate Gaussian Filter Kernel ---
    float filterKernel[MAX_FILTER_SIZE * MAX_FILTER_SIZE];
    generateGaussianKernel(filterKernel, filterSize, sigma);

    printf("Gaussian kernel: %dx%d, sigma=%.1f\n", filterSize, filterSize, sigma);

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

    // --- Load & Build Kernel ---
    FILE* fp = fopen("gaussian_blur.cl", "rb");
    if (!fp) {
        printf("Failed to open gaussian_blur.cl\n");
        return EXIT_FAILURE;
    }
    fseek(fp, 0, SEEK_END);
    size_t sourceSize = ftell(fp);
    rewind(fp);
    char* source = (char*)malloc(sourceSize + 1);
    fread(source, 1, sourceSize, fp);
    source[sourceSize] = '\0';
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, &err);
    checkError(err, "clCreateProgramWithSource");
    free(source);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* buildLog = (char*)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
        printf("Build error:\n%s\n", buildLog);
        free(buildLog);
        return EXIT_FAILURE;
    }
    printf("Kernel compiled successfully\n");

    cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &err);
    checkError(err, "clCreateKernel");

    // --- Set Kernel Arguments ---
    int widthArg = (int)image.width;
    int heightArg = (int)image.height;
    int bppArg = (int)bytesPerPixel;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &widthArg);
    clSetKernelArg(kernel, 4, sizeof(int), &heightArg);
    clSetKernelArg(kernel, 5, sizeof(int), &filterSize);
    clSetKernelArg(kernel, 6, sizeof(int), &bppArg);

    // --- Execute Kernel with 2D NDRange ---
    size_t globalWorkSize[2] = { image.width, image.height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    checkError(err, "clEnqueueNDRangeKernel");

    clFinish(queue);
    printf("Kernel executed (%ux%u work items)\n", image.width, image.height);

    // --- Read Back Result ---
    unsigned char* outputData = (unsigned char*)malloc(imageSize);
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imageSize, outputData, 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer");

    // --- Save Output Image ---
    if (!saveBMP(outputFile, outputData, image.width, image.height, bytesPerPixel)) {
        printf("Failed to save output image: %s\n", outputFile);
        free(outputData);
        return EXIT_FAILURE;
    }
    free(outputData);
    printf("Output saved to: %s\n", outputFile);

    // cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(filterBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(inputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(platforms);

    return EXIT_SUCCESS;
}
