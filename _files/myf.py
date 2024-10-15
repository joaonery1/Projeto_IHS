import os
import numpy as np
import pyopencl as cl
from PIL import Image
import matplotlib.pyplot as plt

# Habilita a saída do compilador OpenCL
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

# Carregar a imagem
image = Image.open('input.png').convert('L')  # Converte para escala de cinza
image = np.array(image, dtype=np.uint8)

# Parâmetros da imagem
height, width = image.shape

# Definindo o kernel de convolução (filtro de média 5x5)
kernel = np.ones((5, 5), dtype=np.float32) / 25.0

# Configurando OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Criando buffers
image_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image.nbytes)
kernel_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)

# Código do kernel
kernel_code = """
__kernel void convolve(__global const uchar* input, __global uchar* output, 
                       const int width, const int height, 
                       __global const float* filter_kernel) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Verifica se está dentro dos limites da imagem
    if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        float sum = 0.0f;
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                sum += input[(y + ky) * width + (x + kx)] * filter_kernel[(ky + 2) * 5 + (kx + 2)];
            }
        }
        output[y * width + x] = (uchar)clamp(sum, 0.0f, 255.0f);
    }
}
"""

# Compilando o kernel
program = cl.Program(context, kernel_code).build()

# Definindo as dimensões do trabalho
global_work_size = (width, height)

# Executando o kernel
program.convolve(queue, global_work_size, None, image_buf, output_buf, 
                 np.int32(width), np.int32(height), kernel_buf)

# Lendo o resultado
output_image = np.empty_like(image)
cl.enqueue_copy(queue, output_image, output_buf)

# Visualizando as imagens
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagem Convoluída")
plt.imshow(output_image, cmap='gray')
plt.axis('off')

plt.show()
