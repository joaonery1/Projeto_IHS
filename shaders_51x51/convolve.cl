__kernel void convolve(__global const uchar* input,
                       __global uchar* output,
                       const int width,
                       const int height,
                       __global const float* filter_kernel) {
    int half_size = 25; // Para um kernel 51x51
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    float sum = 0.0f;

    if (x < width && y < height) {
        for (int ky = -half_size; ky <= half_size; ky++) {
            for (int kx = -half_size; kx <= half_size; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    // Acesse o pixel da imagem de entrada
                    float pixel_value = (float)input[ny * width + nx];
                    // Adicione a contribuição do filtro
                    sum += pixel_value * filter_kernel[(ky + half_size) * 51 + (kx + half_size)];
                }
            }
        }
        // Clamping e atribuição ao output
        output[y * width + x] = (uchar)(max(0.0f, min(sum, 255.0f)));
    }
}
