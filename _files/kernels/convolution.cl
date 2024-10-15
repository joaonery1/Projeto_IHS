__kernel void convolve(__global const uchar* input,
                       __global uchar* output,
                       const int width,
                       const int height,
                       __global const float* filter_kernel) {
    int half_size = 1; // Para um kernel 3x3
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Verifica se está dentro dos limites da imagem
    if (x >= half_size && x < width - half_size && y >= half_size && y < height - half_size) {
        float sum = 0.0f;

        // Percorre o elemento estruturante
        for (int i = -half_size; i <= half_size; ++i) {
            for (int j = -half_size; j <= half_size; ++j) {
                sum += input[(y + i) * width + (x + j)] * filter_kernel[(i + half_size) * 3 + (j + half_size)];
            }
        }

        // Clampeia e escreve o valor de saída
        output[y * width + x] = (uchar)clamp(sum, 0.0f, 255.0f);
    }
}

// Função clamp
uchar clamp(float value) {
    if (value < 0.0f) return 0;
    if (value > 255.0f) return 255;
    return (uchar)value;
}
