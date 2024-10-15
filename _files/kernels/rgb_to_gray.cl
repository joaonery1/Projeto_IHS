__kernel void rgb_to_gray(__global const uchar4* input, __global uchar* output,
                          const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Verifica se está dentro dos limites da imagem
    if (x < width && y < height) {
        uchar4 pixel = input[y * width + x];
        
        // Calcular o valor de cinza usando coeficientes
        uchar gray_value = (uchar)(0.2989f * pixel.x + 0.5870f * pixel.y + 0.1140f * pixel.z);
        
        output[y * width + x] = gray_value; // Armazenar valor de cinza
    }
}
