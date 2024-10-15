__kernel void subtract(__global const uchar* image1,
                       __global const uchar* image2,
                       __global uchar* output,
                       const int width,
                       const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int index = y * width + x;
        int result = (int)image1[index] - (int)image2[index];
        output[index] = (uchar)clamp(result, 0, 255);
    }
}
