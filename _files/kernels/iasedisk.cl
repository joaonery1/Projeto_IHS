__kernel void iasedisk(__global int* output, int r, int h, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        // Verifica se está dentro do círculo definido pela métrica euclidiana
        float distance = sqrt((x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2));
        if (distance <= (r + 0.5)) {
            output[y * width + x] = h + (int)(sqrt((r + 0.5) * (r + 0.5) - distance * distance));
        } else {
            output[y * width + x] = 0;
        }
    }
}
