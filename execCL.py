import os
import numpy as np
import pyopencl as cl
from PIL import Image
import matplotlib.pyplot as plt
import time

# Habilita a saída do compilador OpenCL
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

# Função para listar plataformas e dispositivos disponíveis
def list_platforms_devices():
    platforms = cl.get_platforms()
    if not platforms:
        print("Nenhuma plataforma OpenCL encontrada.")
        exit(1)
    
    print("Plataformas e Dispositivos OpenCL disponíveis:\n")
    for i, platform in enumerate(platforms):
        print(f"Plataforma {i}: {platform.name}")
        devices = platform.get_devices()
        if not devices:
            print("  Nenhum dispositivo encontrado nesta plataforma.")
            continue
        for j, device in enumerate(devices):
            print(f"  Dispositivo {j}: {device.name}")
            print(f"    Tipo de dispositivo: {cl.device_type.to_string(device.type)}")
            print(f"    Memória global: {device.global_mem_size / (1024**3):.2f} GB")
            print(f"    Memória local: {device.local_mem_size / 1024:.2f} KB")
            print(f"    Versão OpenCL: {device.opencl_c_version}")
        print()
    
    # Escolher a plataforma
    while True:
        try:
            platform_index = int(input(f"Escolha a Plataforma [0-{len(platforms)-1}]: "))
            if 0 <= platform_index < len(platforms):
                break
            else:
                print(f"Por favor, insira um número entre 0 e {len(platforms)-1}.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número inteiro.")
    
    selected_platform = platforms[platform_index]
    devices = selected_platform.get_devices()
    
    if not devices:
        print("Nenhum dispositivo encontrado na plataforma selecionada.")
        exit(1)
    
    # Escolher o dispositivo
    while True:
        try:
            device_index = int(input(f"Escolha o Dispositivo [0-{len(devices)-1}]: "))
            if 0 <= device_index < len(devices):
                break
            else:
                print(f"Por favor, insira um número entre 0 e {len(devices)-1}.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número inteiro.")
    
    selected_device = devices[device_index]
    
    print(f"\nPlataforma selecionada: {selected_platform.name}")
    print(f"Dispositivo selecionado: {selected_device.name}\n")
    
    return selected_platform, selected_device

# Função para carregar o código do kernel a partir de um arquivo
def load_kernel_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()

# Função para configurar o OpenCL
def setup_opencl(image, platform, device):
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    # Criando buffers
    image_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
    output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image.nbytes)
    
    return context, queue, image_buf, output_buf

# Função para executar a dilatação
def dilate_image(image, structuring_element, platform, device):
    height, width = image.shape

    context, queue, image_buf, output_buf = setup_opencl(image, platform, device)
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
def erode_image(image, structuring_element, platform, device):
    height, width = image.shape

    context, queue, image_buf, output_buf = setup_opencl(image, platform, device)
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
def convolve_image(image, filter_kernel, platform, device):
    height, width = image.shape

    context, queue, image_buf, output_buf = setup_opencl(image, platform, device)
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
    average_time = (total_time / num_steps) * 1000
    return average_time

# Função para salvar a imagem
def save_image(image, filename):
    img = Image.fromarray(image)
    img.save(filename)

def main():
    # Listar plataformas e dispositivos e permitir a escolha
    selected_platform, selected_device = list_platforms_devices()
    
    # Carregar a imagem
    image_path = 'input.JPG'
    if not os.path.exists(image_path):
        print(f"Arquivo de imagem '{image_path}' não encontrado.")
        exit(1)
    
    image = Image.open(image_path).convert('L')  # Converte para escala de cinza
    image = np.array(image, dtype=np.uint8)
    
    # Definindo um elemento estruturante 51x51
    structuring_element = np.ones((51, 51), dtype=np.float32)
    
    # Realizar benchmarks
    num_steps = 10  # Reduzi para 10 para acelerar o processo; ajuste conforme necessário
    
    print("Realizando benchmarks...")
    dilate_time = benchmark_function(dilate_image, image, structuring_element, selected_platform, selected_device, num_steps=num_steps)
    erode_time = benchmark_function(erode_image, image, structuring_element, selected_platform, selected_device, num_steps=num_steps)
    
    # Para a convolução, precisa da imagem erodida
    print("Erosão concluída. Preparando para convolução...")
    final_image = erode_image(image, structuring_element, selected_platform, selected_device)
    
    filter_kernel = np.ones((51, 51), dtype=np.float32) / 2601.0  # Normalizando
    convolve_time = benchmark_function(convolve_image, final_image, filter_kernel, selected_platform, selected_device, num_steps=num_steps)
    
    # Exibir resultados
    print(f"\nTempo médio de dilatação: {dilate_time:.6f} ms")
    print(f"Tempo médio de erosão: {erode_time:.6f} ms")
    print(f"Tempo médio de convolução: {convolve_time:.6f} ms")
    
    # Processar e salvar as imagens finais
    print("\nProcessando imagens finais...")
    
    dilated_image = dilate_image(image, structuring_element, selected_platform, selected_device)
    plt.figure(figsize=(8, 6))
    plt.title("Imagem Dilatação")
    plt.imshow(dilated_image, cmap='gray')
    plt.axis('off')
    plt.show()
    save_image(dilated_image, "out/dilate.png") 
    
    eroded_image = erode_image(image, structuring_element, selected_platform, selected_device)
    plt.figure(figsize=(8, 6))
    plt.title("Imagem Erosão")
    plt.imshow(eroded_image, cmap='gray')
    plt.axis('off')
    plt.show()
    save_image(eroded_image, "out/erode.png") 
    
    final_image = convolve_image(eroded_image, filter_kernel, selected_platform, selected_device)
    
    plt.figure(figsize=(8, 6))
    plt.title("Imagem Final (Convolução)")
    plt.imshow(final_image, cmap='gray')
    plt.axis('off')
    plt.show()
    save_image(final_image, "out/convolve.png") 
    
    print("Processamento concluído e imagens salvas na pasta 'out/'.")

if __name__ == "__main__":
    main()
