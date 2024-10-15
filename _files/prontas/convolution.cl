__kernel void convolve(__global const uchar* image,
                       __global uchar* output,
                       const int width,
                       const int height,
                       __global const float* se) {
    int half_size = 10; // Para um kernel 21x21
    int x = get_global_id(0);
    int y = get_global_id(1);

    float sum = 0.0f;
    float norm = 0.0f; // Para normalizar a soma

    // Percorre o elemento estruturante
    for (int i = -half_size; i <= half_size; ++i) {
        for (int j = -half_size; j <= half_size; ++j) {
            int nx = x + j;
            int ny = y + i;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float weight = se[i + half_size + (j + half_size) * 21];
                sum += image[nx + ny * width] * weight;
                norm += weight; // Acumula a normalização
            }
        }
    }

    if (norm > 0) {
        output[x + y * width] = clamp(sum / norm, 0.0f, 255.0f); // Normaliza a soma
    } else {
        output[x + y * width] = 0; // Se não há peso, define para zero
    }
}
