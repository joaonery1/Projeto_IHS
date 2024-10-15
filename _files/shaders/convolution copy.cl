__kernel void convolve(__global const uchar* input,
                       __global uchar* output,
                       const int width,
                       const int height,
                       __global const float* filter_kernel) {
    int half_size = 1; // Para um kernel 3x3
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = -half_size; ky <= half_size; ky++) {
            for (int kx = -half_size; kx <= half_size; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx] * filter_kernel[(ky + half_size) * 3 + (kx + half_size)];
                }
            }
        }
        output[y * width + x] = (uchar)clamp(sum, 0.0f, 255.0f);
    }
}
