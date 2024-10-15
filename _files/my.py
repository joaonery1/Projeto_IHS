import os
import numpy as np
import pyopencl as cl
from PIL import Image
import matplotlib.pyplot as plt

# Habilita a saída do compilador OpenCL
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

# Função para carregar o código do kernel a partir de um arquivo
def load_kernel_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()

# Carregar a imagem
image = Image.open('input.png').convert('L')  # Converte para escala de cinza
image = np.array(image, dtype=np.uint8)

# Parâmetros da imagem
height, width = image.shape

# Definindo um elemento estruturante 3x3
structuring_element = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]], dtype=np.float32)

# Configurando OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Criando buffers
image_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image.nbytes)
se_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=structuring_element)

# Carregar o código do kernel de erosão
kernel_code = load_kernel_from_file('shaders/erode.cl')

# Compilando o kernel
program = cl.Program(context, kernel_code).build()

# Definindo as dimensões do trabalho
global_work_size = (width, height)

# Executando o kernel de erosão
program.erode(queue, global_work_size, None, image_buf, output_buf, 
              np.int32(width), np.int32(height), se_buf)

# Lendo o resultado
output_image = np.empty_like(image)
cl.enqueue_copy(queue, output_image, output_buf)

# Visualizando as imagens
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagem Erosão")
plt.imshow(output_image, cmap='gray')
plt.axis('off')

plt.show()
