import os
import numpy as np
import pyopencl as cl
from PIL import Image
import matplotlib.pyplot as plt
import time

# Habilita a saída do compilador OpenCL
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

# Função para carregar o código do kernel a partir de um arquivo
def load_kernel_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()

# Função para configurar o OpenCL
def setup_opencl(image):
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Criando buffers
    image_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
    output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image.nbytes)

    return context, queue, image_buf, output_buf

# Função para executar a dilatação
def dilate_image(image, structuring_element):
    height, width = image.shape

    context, queue, image_buf, output_buf = setup_opencl(image)
    se_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=structuring_element)

    # Carregar o código do kernel de dilatação
    kernel_code = load_kernel_from_file('shaders_51x51/dilatacao.cl')
    program = cl.Program(context, kernel_code).build()

    # Definindo as dimensões do trabalho
    global_work_size = (width, height)

    # Executando o kernel de dilatação
    program.dilate(queue, global_work_size, None, image_buf, output_buf, 
                   np.int32(width), np.int32(height), se_buf)

    # Lendo o resultado
    dilated_image = np.empty_like(image)
    cl.enqueue_copy(queue, dilated_image, output_buf)

    return dilated_image

# Função para executar a erosão
def erode_image(image, structuring_element):
    height, width = image.shape

    context, queue, image_buf, output_buf = setup_opencl(image)
    se_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=structuring_element)

    # Carregar o código do kernel de erosão
    kernel_code = load_kernel_from_file('shaders_51x51/erode.cl')
    program = cl.Program(context, kernel_code).build()

    # Definindo as dimensões do trabalho
    global_work_size = (width, height)

    # Executando o kernel de erosão
    program.erode(queue, global_work_size, None, image_buf, output_buf, 
                  np.int32(width), np.int32(height), se_buf)

    # Lendo o resultado
    eroded_image = np.empty_like(image)
    cl.enqueue_copy(queue, eroded_image, output_buf)

    return eroded_image

# Função para executar a convolução
def convolve_image(image, filter_kernel):
    height, width = image.shape

    context, queue, image_buf, output_buf = setup_opencl(image)
    filter_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=filter_kernel)

    # Carregar o código do kernel de convolução
    kernel_code = load_kernel_from_file('shaders_51x51/convolve.cl')
    program = cl.Program(context, kernel_code).build()

    # Definindo as dimensões do trabalho
    global_work_size = (width, height)

    # Executando o kernel de convolução
    program.convolve(queue, global_work_size, None, image_buf, output_buf, 
                     np.int32(width), np.int32(height), filter_buf)

    # Lendo o resultado
    convolved_image = np.empty_like(image)
    cl.enqueue_copy(queue, convolved_image, output_buf)

    return convolved_image

# Função para benchmark
def benchmark_function(func, *args, num_steps=1000):
    total_time = 0
    for _ in range(num_steps):
        start_time = time.time()
        func(*args)
        end_time = time.time()
        total_time += (end_time - start_time)
    average_time = total_time / num_steps
    return average_time

# Carregar a imagem
image = Image.open('input.JPG').convert('L')  # Converte para escala de cinza
image = np.array(image, dtype=np.uint8)

# Definindo um elemento estruturante 51x51
structuring_element = np.ones((51, 51), dtype=np.float32)

# Realizar benchmarks
num_steps = 10

dilate_time = benchmark_function(dilate_image, image, structuring_element, num_steps=num_steps)
erode_time = benchmark_function(erode_image, image, structuring_element, num_steps=num_steps)
final_image = erode_image(image, structuring_element)  # Para a convolução, precisa da imagem erodida

filter_kernel = np.ones((51, 51), dtype=np.float32) / 2601.0  # Normalizando
convolve_time = benchmark_function(convolve_image, final_image, filter_kernel, num_steps=num_steps)

# Exibir resultados
print(f"Tempo médio de dilatação: {dilate_time:.6f} segundos")
print(f"Tempo médio de erosão: {erode_time:.6f} segundos")
print(f"Tempo médio de convolução: {convolve_time:.6f} segundos")

# Visualizar os resultados
dilated_image = dilate_image(image, structuring_element)
plt.figure(figsize=(8, 6))
plt.title("Imagem Dilatação")
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')
plt.show()

eroded_image = erode_image(image, structuring_element)
plt.figure(figsize=(8, 6))
plt.title("Imagem Erosão")
plt.imshow(eroded_image, cmap='gray')
plt.axis('off')
plt.show()

# Aplicando convolução na imagem erodida
final_image = convolve_image(eroded_image, filter_kernel)

# Visualizando a imagem final após convolução
plt.figure(figsize=(8, 6))
plt.title("Imagem Final (Convolução)")
plt.imshow(final_image, cmap='gray')
plt.axis('off')
plt.show()
