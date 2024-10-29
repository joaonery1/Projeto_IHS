# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import my
import cv2
import numpy as np
import time as t
from datetime import datetime

def close(im, kernel, iterations=1):
    imdil = cv2.dilate(im, kernel, iterations)
    result = cv2.erode(imdil, kernel, iterations)
    return result

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def blackhat(im, kernel, iterations=1):
    result = close(im, kernel, iterations)
    return result

def smooth(im, diam=3):
    result = cv2.GaussianBlur(im, (diam, diam), 0)
    return result

def image_equal(im0, im1):
    return (sum(sum(im0 != im1)) == 0)

def reconstruct(im):
    kernel = np.ones((3, 3), np.uint8)
    imero = cv2.erode(im, kernel)
    c = 0
    imt0 = imero
    imt1 = cv2.dilate(imt0, kernel)
    is_equal = image_equal(imt0, imt1)
    while (not is_equal):
        print(c)
        imt0 = imt1
        imdil = cv2.dilate(imt0, kernel)
        imt1 = np.minimum(imdil, im)
        is_equal = image_equal(imt0, imt1)
        c = c + 1
    return imt1

msg = ""
media = 0.0
nSteps = 100
total = 0.0
TEST1 = False
TEST2 = False
TEST3 = True

if __name__ == "__main__":
    filename = "input.JPG"
    img = my.imread(filename)
    imgray = my.imreadgray(filename)

    if (TEST3):
        kernel_t = cv2.getGaussianKernel(51, 1)

        # Kernel de 51x51
        kernel_size = 51
        kernel_51x51 = np.ones((kernel_size, kernel_size), np.uint8)

        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #my.imshow(imgray)
        
        # Salvando a imagem em escala de cinza
        #cv2.imwrite('out_cv/output_gray.jpg', imgray)

        t0 = datetime.now()
        for i in range(nSteps):
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t1 = datetime.now()
        t = t1 - t0
        media = (t.total_seconds() * 1000) / nSteps
        #msg += f"Tempo médio de {nSteps} execuções do método Rgb2Gray: {media} ms\n"
        total += media

        # Suavização
        imsmooth = cv2.sepFilter2D(imgray, -1, kernel_t, kernel_t)

        # Salvando a imagem suavizada
        cv2.imwrite('out_cv/convolve.jpg', imsmooth)

        # Runtime para suavização
        t0 = datetime.now()
        for i in range(nSteps):
            imsmooth = cv2.sepFilter2D(imgray, -1, kernel_t, kernel_t)
        t1 = datetime.now()
        t = t1 - t0
        media = (t.total_seconds() * 1000) / nSteps
        msg += f"Tempo médio de {nSteps} execuções do método Convolution: {media} ms\n"
        total += media

        # Dilatação e Erosão
        imdil = cv2.dilate(imsmooth, kernel_51x51, 1)

        # Salvando a imagem dilatada
        cv2.imwrite('out_cv/dilate.jpg', imdil)

        # Runtime para dilatação
        t0 = datetime.now()
        for i in range(nSteps):
            imdil = cv2.dilate(imsmooth, kernel_51x51, 1)
        t1 = datetime.now()
        t = t1 - t0
        media = (t.total_seconds() * 1000) / nSteps
        msg += f"Tempo médio de {nSteps} execuções do método Dilate: {media} ms\n"
        total += media

        imerode = cv2.erode(imdil, kernel_51x51, 1)

        # Salvando a imagem erodida
        cv2.imwrite('out_cv/erode.jpg', imerode)

        # Runtime para erosão
        t0 = datetime.now()
        for i in range(nSteps):
            imerode = cv2.erode(imdil, kernel_51x51, 1)
        t1 = datetime.now()
        t = t1 - t0
        media = (t.total_seconds() * 1000) / nSteps
        msg += f"Tempo médio de {nSteps} execuções do método Erode: {media} ms\n"
        total += media

print("-------------------------------------------------------------")            
print(msg)
print("-------------------------------------------------------------")
print("Valor total médio: " + str(total))
