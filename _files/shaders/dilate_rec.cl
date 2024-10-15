__kernel void dilation_reconstruction(__global const uchar* image,
                                      __global const uchar* marker,
                                      __global uchar* output,
                                      const int width,
                                      const int height,
                                      __global const float* se) {
    int half_size = 1; // Para um kernel 3x3
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Inicializa o valor de saída
    uchar max_value = 0;

    // Verifica se a posição do marcador é maior que 0
    if (marker[y * width + x] > 0) {
        // Percorre o elemento estruturante
        for (int i = -half_size; i <= half_size; ++i) {
            for (int j = -half_size; j <= half_size; ++j) {
                int nx = x + j;
                int ny = y + i;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    // Se o elemento estruturante for positivo, obtém o valor do input
                    if (se[(i + half_size) * 3 + (j + half_size)] > 0) {
                        max_value = max(max_value, image[nx + ny * width]);
                    }
                }
            }
        }
        // Define o valor de saída
        output[y * width + x] = max_value;
    } else {
        // Se o marcador não for maior que 0, mantém o valor original
        output[y * width + x] = image[y * width + x];
    }
}
