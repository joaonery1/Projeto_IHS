__kernel void erode(__global const uchar* image,
                    __global uchar* output,
                    const int width,
                    const int height,
                    __global const float* se) {
    int half_size = 10; // Para um kernel 21x21
    int x = get_global_id(0);
    int y = get_global_id(1);

    uchar min_value = 255; // Inicializa com o valor máximo

    // Percorre o elemento estruturante
    for (int i = -half_size; i <= half_size; ++i) {
        for (int j = -half_size; j <= half_size; ++j) {
            int nx = x + j;
            int ny = y + i;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (se[i + half_size + (j + half_size) * 21] > 0) {
                    min_value = min(min_value, image[nx + ny * width]);
                }
            }
        }
    }

    output[x + y * width] = min_value;
}
