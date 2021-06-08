from PIL import Image, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(int(time.time()))
def noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def readImg2array(file, threshold=60):
    size = file[1]
    pilIN = Image.open(file[0]).convert(mode="L")
    pilIN = pilIN.resize((size[1], size[0]))
    imgArray = np.asarray(pilIN, dtype=np.uint8)
    temp = np.zeros(imgArray.shape, dtype=np.float64)
    temp[imgArray > threshold] = 1
    temp[temp == 0] = -1
    return temp

def array2img(data, outFile=None):
    y = np.zeros(data.shape, dtype=np.uint8)
    y[data == 1] = 255
    y[data == -1] = 0
    img = Image.fromarray(y, mode="L")
    if outFile is not None:
        img.save(outFile)
    return img

def mat2vec(x):
    m = x.shape[0] * x.shape[1]
    tmp1 = np.zeros(m)
    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i, j]
            c += 1
    return tmp1

def train_and_test_data():
    percent = [0.1, 0.3, 0.6]
    font_size = [64, 32, 16]
    for f in range(len(font_size)):
        font = ImageFont.truetype("arial.ttf", font_size[f])
        for char in "ABCDEFGHIJ":
            for p in range(len(percent)):
                im = Image.Image()._new(font.getmask(char))
                im.save(f"image/{char}_{font_size[f]}.bmp")
                matrix = readImg2array((f"image/{char}_{font_size[f]}.bmp", im.size))
                noisy_matrix = noise(matrix, percent[p])
                array2img(noisy_matrix, f"noisy_image/{char}_{font_size[f]}_{percent[p]}_noisy.bmp")

train_and_test_data()