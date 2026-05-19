__kernel void gaussian_blur(
    __global const unsigned char* input,
    __global unsigned char* output,
    __global const float* filter,
    const int width,
    const int height,
    const int filterSize,
    const int bytesPerPixel)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int radius = filterSize / 2;

    float channels[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int fy = 0; fy < filterSize; fy++) {
        for (int fx = 0; fx < filterSize; fx++) {
            // sample position with border clamping
            int sampleX = x + fx - radius;
            int sampleY = y + fy - radius;

            // clamp to nearest valid pixel
            if (sampleX < 0) sampleX = 0;
            if (sampleX >= width) sampleX = width - 1;
            if (sampleY < 0) sampleY = 0;
            if (sampleY >= height) sampleY = height - 1;

            int sampleIdx = (sampleY * width + sampleX) * bytesPerPixel;
            float weight = filter[fy * filterSize + fx];

            for (int c = 0; c < bytesPerPixel; c++) {
                channels[c] += (float)input[sampleIdx + c] * weight;
            }
        }
    }

    int outIdx = (y * width + x) * bytesPerPixel;
    for (int c = 0; c < bytesPerPixel; c++) {
        // clamp result to valid range
        float val = channels[c];
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        output[outIdx + c] = (unsigned char)(val + 0.5f);
    }
}
