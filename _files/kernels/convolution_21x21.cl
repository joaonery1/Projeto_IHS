__kernel void convolution_21x21(__global const uchar* input, __global uchar* output,
                                 const int width, const int height, 
                                 __global const float* kernel) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    // Verifica se está dentro dos limites da imagem
    if (x >= 10 && x < width - 10 && y >= 10 && y < height - 10) {
        float sum = 0.0f;
        for (int ky = -10; ky <= 10; ky++) {
            for (int kx = -10; kx <= 10; kx++) {
                sum += input[(y + ky) * width + (x + kx)] * kernel[(ky + 10) * 21 + (kx + 10)];
            }
        }
        output[y * width + x] = (uchar)clamp(sum, 0.0f, 255.0f);
    }
}
