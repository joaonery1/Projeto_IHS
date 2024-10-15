# Projeto de Processamento de Imagens

Este projeto utiliza OpenCV e OpenCL para realizar diversas operações de processamento de imagens. O objetivo é demonstrar o uso dessas bibliotecas para realizar tarefas como suavização, dilatação e erosão em imagens.

## Requisitos

Antes de executar o projeto, você precisa ter as seguintes dependências instaladas:

- Python 3.x
- OpenCV
- NumPy
- [OpenCL](https://www.khronos.org/opencl/)

### Instalação

Você pode instalar as dependências necessárias usando o `pip`. Execute o seguinte comando:

```bash
pip install opencv-python numpy pyopencl
```

## Execução

Para rodar o projeto, você pode usar duas abordagens diferentes: uma utilizando OpenCV e outra utilizando OpenCL.

### Usando OpenCV

Para executar o script com OpenCV, utilize o seguinte comando:

```bash
python3 execCV.py
```

### Usando OpenCL

Para executar o script com OpenCL, utilize o seguinte comando:

```bash
python3 execCL.py
```

## Estrutura do Projeto

- `execCV.py`: Script principal que utiliza OpenCV para processamento de imagens.
- `execCL.py`: Script principal que utiliza OpenCL para processamento de imagens.
- `input.JPG`: Imagem de entrada que será processada.

## Contribuição

Sinta-se à vontade para contribuir com melhorias, correções ou novas funcionalidades. Para contribuir, siga estas etapas:

1. Fork este repositório.
2. Crie uma nova branch (`git checkout -b feature/nome-da-feature`).
3. Faça suas alterações e commit (`git commit -m 'Adiciona nova feature'`).
4. Envie para o repositório remoto (`git push origin feature/nome-da-feature`).
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
