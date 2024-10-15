__kernel void threshold(__global const uchar* image,
                        __global uchar* output,
                        const int width,
                        const int height,
                        const uchar threshold_value) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Verifica se estamos dentro dos limites da imagem
    if (x < width && y < height) {
        int index = y * width + x;
        // Aplica o threshold
        output[index] = (image[index] > threshold_value) ? 255 : 0;
    }
}
