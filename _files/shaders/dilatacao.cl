__kernel void dilate(__global const uchar* input, __global uchar* output, 
                     const int width, const int height, 
                     __global const float* structuring_element) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Verifica se está dentro dos limites da imagem
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        uchar max_value = 0; // Inicializa com o valor mínimo

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                // Aplica o elemento estruturante
                if (structuring_element[(ky + 1) * 3 + (kx + 1)] > 0) {
                    uchar pixel_value = input[(y + ky) * width + (x + kx)];
                    if (pixel_value > max_value) {
                        max_value = pixel_value; // Encontra o máximo
                    }
                }
            }
        }
        output[y * width + x] = max_value; // Atribui o valor máximo encontrado
    }
}
