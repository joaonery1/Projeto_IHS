__kernel void dilate(__global const uchar* input, __global uchar* output, 
                     const int width, const int height, 
                     __global const float* structuring_element) {
    int half_size = 25; // Para um kernel 51x51
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Verifica se está dentro dos limites da imagem
    if (x >= half_size && x < width - half_size && y >= half_size && y < height - half_size) {
        uchar max_value = 0; // Inicializa com o valor mínimo

        for (int ky = -half_size; ky <= half_size; ky++) {
            for (int kx = -half_size; kx <= half_size; kx++) {
                // Aplica o elemento estruturante
                if (structuring_element[(ky + half_size) * 51 + (kx + half_size)] > 0) {
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
