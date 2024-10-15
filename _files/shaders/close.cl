__kernel void close(__global const uchar* image,
                    __global uchar* output,
                    const int width,
                    const int height,
                    __global const float* se) {
    int half_size = 1; // Para um kernel 3x3
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    // Etapa de erosão
    uchar min_value = 255; // Inicializa com o valor máximo

    // Erosão
    for (int i = -half_size; i <= half_size; ++i) {
        for (int j = -half_size; j <= half_size; ++j) {
            int nx = x + j;
            int ny = y + i;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                // Verifica se o elemento estruturante é positivo
                if (se[(i + half_size) * 3 + (j + half_size)] > 0) {
                    min_value = min(min_value, image[nx + ny * width]);
                }
            }
        }
    }
    
    // Armazena o resultado da erosão
    uchar eroded_value = min_value;

    // Etapa de dilatação
    uchar max_value = 0;

    // Dilatação
    for (int i = -half_size; i <= half_size; ++i) {
        for (int j = -half_size; j <= half_size; ++j) {
            int nx = x + j;
            int ny = y + i;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                // Se o elemento estruturante for positivo, obtém o valor da erosão
                if (se[(i + half_size) * 3 + (j + half_size)] > 0) {
                    max_value = max(max_value, eroded_value);
                }
            }
        }
    }

    output[y * width + x] = max_value;
}
