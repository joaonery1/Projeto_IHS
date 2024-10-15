__kernel void dilate(__global const uchar* image,
                     __global uchar* output,
                     const int width,
                     const int height,
                     __global const float* se) {
    int half_size = 10; // Para um kernel 21x21
    int x = get_global_id(0);
    int y = get_global_id(1);

    uchar max_value = 0;

    // Percorre o elemento estruturante
    for (int i = -half_size; i <= half_size; ++i) {
        for (int j = -half_size; j <= half_size; ++j) {
            int nx = x + j;
            int ny = y + i;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (se[i + half_size + (j + half_size) * 21] > 0) {
                    max_value = max(max_value, image[nx + ny * width]);
                }
            }
        }
    }

    output[x + y * width] = max_value;
}
