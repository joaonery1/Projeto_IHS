__kernel void thinning(__global const uchar* input, __global uchar* output,
                       const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    // Inicializa a saída igual à entrada
    output[y * width + x] = input[y * width + x];

    // Verifica se está dentro dos limites da imagem
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // O pixel a ser processado
        uchar center = input[y * width + x];

        // Apenas continue se o pixel central for branco (255)
        if (center == 255) {
            // Máscara para verificar vizinhos
            int count = 0;

            // Contar os vizinhos
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    if (input[(y + ky) * width + (x + kx)] == 255) {
                        count++;
                    }
                }
            }

            // Aplicar regras de afinamento
            if (count > 2) {
                // Defina a condição para apagar o pixel
                output[y * width + x] = 0; // Apagar o pixel
            }
        }
    }
}
