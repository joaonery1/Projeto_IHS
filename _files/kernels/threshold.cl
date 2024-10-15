__kernel void threshold(__global const uchar* image,
                        __global uchar* output,
                        const int width,
                        const int height,
                        const uchar threshold_low,
                        const uchar threshold_high) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = x + y * width;
    uchar pixel = image[index];

    // Aplica o threshold
    if (pixel < threshold_low) {
        output[index] = 0;  // Preto
    } else if (pixel > threshold_high) {
        output[index] = 255; // Branco
    } else {
        output[index] = pixel; // Valor original
    }
}
