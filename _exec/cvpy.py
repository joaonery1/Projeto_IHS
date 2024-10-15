import cv2
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

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

# Carregar a imagem usando OpenCV
image = cv2.imread('input.JPG', cv2.IMREAD_GRAYSCALE)

# Definindo um elemento estruturante 51x51
structuring_element = np.ones((51, 51), dtype=np.uint8)

# Função para dilatação usando OpenCV
def dilate_image_opencv(image, structuring_element):
    return cv2.dilate(image, structuring_element)

# Função para erosão usando OpenCV
def erode_image_opencv(image, structuring_element):
    return cv2.erode(image, structuring_element)

# Função para convolução usando OpenCV
def convolve_image_opencv(image, filter_kernel):
    return cv2.filter2D(image, -1, filter_kernel)

# Realizar benchmarks com OpenCV
num_steps = 1000

dilate_time_opencv = benchmark_function(dilate_image_opencv, image, structuring_element, num_steps=num_steps)
erode_time_opencv = benchmark_function(erode_image_opencv, image, structuring_element, num_steps=num_steps)
final_image_opencv = erode_image_opencv(image, structuring_element)  # Para a convolução
filter_kernel = np.array([[1/9, 1/9, 1/9],
                          [1/9, 1/9, 1/9],
                          [1/9, 1/9, 1/9]], dtype=np.float32)  # Kernel 3x3

convolve_time_opencv = benchmark_function(convolve_image_opencv, final_image_opencv, filter_kernel, num_steps=num_steps)

# Exibir resultados de tempo para OpenCV
print(f"Tempo médio de dilatação (OpenCV): {dilate_time_opencv:.6f} segundos")
print(f"Tempo médio de erosão (OpenCV): {erode_time_opencv:.6f} segundos")
print(f"Tempo médio de convolução (OpenCV): {convolve_time_opencv:.6f} segundos")

# Visualizar os resultados de OpenCV
dilated_image_opencv = dilate_image_opencv(image, structuring_element)
plt.figure(figsize=(8, 6))
plt.title("Imagem Dilatação (OpenCV)")
plt.imshow(dilated_image_opencv, cmap='gray')
plt.axis('off')
plt.show()

eroded_image_opencv = erode_image_opencv(image, structuring_element)
plt.figure(figsize=(8, 6))
plt.title("Imagem Erosão (OpenCV)")
plt.imshow(eroded_image_opencv, cmap='gray')
plt.axis('off')
plt.show()

# Aplicando convolução na imagem erodida
final_image_opencv = convolve_image_opencv(eroded_image_opencv, filter_kernel)

# Visualizando a imagem final após convolução
plt.figure(figsize=(8, 6))
plt.title("Imagem Final (Convolução OpenCV)")
plt.imshow(final_image_opencv, cmap='gray')
plt.axis('off')
plt.show()
