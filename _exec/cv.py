import os
import numpy as np
import cv2  # Importa o OpenCV
from PIL import Image
import matplotlib.pyplot as plt
import time

# Função para executar a dilatação
def dilate_image(image, structuring_element):
    return cv2.dilate(image, structuring_element)

# Função para executar a erosão
def erode_image(image, structuring_element):
    return cv2.erode(image, structuring_element)

# Função para executar a convolução
def convolve_image(image, filter_kernel):
    return cv2.filter2D(image, -1, filter_kernel.reshape((3, 3)))

# Função para benchmark
def benchmark_function(func, *args, num_steps=10):
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
structuring_element = np.ones((51, 51), dtype=np.uint8)

# Realizar benchmarks
num_steps = 10

dilate_time = benchmark_function(dilate_image, image, structuring_element, num_steps=num_steps)
erode_time = benchmark_function(erode_image, image, structuring_element, num_steps=num_steps)

# Para a convolução, precisa da imagem erodida
eroded_image = erode_image(image, structuring_element)
filter_kernel = np.array([[1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9]], dtype=np.float32)  # Kernel de 3x3
convolve_time = benchmark_function(convolve_image, eroded_image, filter_kernel, num_steps=num_steps)

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
