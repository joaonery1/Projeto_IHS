__kernel void subtract(__global const uchar* image1,
                       __global const uchar* image2,
                       __global uchar* output,
                       const int width,
                       const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Realiza a subtração pixel a pixel
    int index = x + y * width;
    uchar pixel1 = image1[index];
    uchar pixel2 = image2[index];
    
    // Garante que o resultado não seja negativo
    output[index] = (pixel1 > pixel2) ? (pixel1 - pixel2) : 0;
}
