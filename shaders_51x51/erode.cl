__kernel void erode(__global const uchar* image,
                    __global uchar* output,
                    const int width,
                    const int height,
                    __global const float* se) {
    int half_size = 25; // Para um kernel 51x51
    int x = get_global_id(0);
    int y = get_global_id(1);

    uchar min_value = 255; // Inicializa com o valor máximo
    bool fits = true; // Para verificar se o elemento estruturante se encaixa

    // Percorre o elemento estruturante
    for (int i = -half_size; i <= half_size; ++i) {
        for (int j = -half_size; j <= half_size; ++j) {
            int nx = x + j;
            int ny = y + i;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                fits = false; // Se o pixel estiver fora dos limites
            } else if (se[(i + half_size) * 51 + (j + half_size)] > 0) {
                min_value = min(min_value, image[nx + ny * width]);
            }
        }
    }

    // Se o elemento estruturante se encaixar, atribui o valor mínimo
    if (fits) {
        output[x + y * width] = min_value;
    } else {
        output[x + y * width] = 0; // Ou qualquer valor que você queira para bordas
    }
}
