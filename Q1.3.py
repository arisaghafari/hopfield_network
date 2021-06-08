import os
import numpy as np
from PIL import Image, ImageFont
import time

global_size = (50, 50)
network_size = 50 * 50

class Hopfield(object):
  def __init__(self, paths):
    self.weight = np.zeros((network_size, network_size))
    for path in paths:
        image_matrix, image_size = readImg2array(f'image/{path}')
        image_matrix = image_matrix.flatten()
        self.weight_update(image_matrix)

  def weight_update(self, vector):
    for i in range(len(vector)):
        for j in range(i,len(vector)):
          if i == j:
            self.weight[i,j] = -50
          else:
            self.weight[i,j] += vector[i] * vector[j]
            self.weight[j,i] = self.weight[i,j]

  def evaluation(self, paths):
      for path in paths:
          print("path : ", path)
          image_matrix, image_size = readImg2array(f'noisy_image/{path}')
          image_matrix = image_matrix.flatten()
          output = self.update(image_matrix)
          output = np.reshape(output, global_size)
          output_resize = Image.fromarray(output, mode='L').resize(image_size)
          output_resize.save('result/' + path)

  def update(self, input):
      output = []
      t = 0
      for row in range(len(self.weight)):
        temp = 0
        for column in range(len(self.weight)):
            if row == column:
              continue
            temp += self.weight[row, column] * input[column]
        if temp >= 0:
            t = 255
        else:
            t = 0
        output.append(t)
      return np.array(output, dtype='uint8')

def readImg2array(file, threshold=60):
    pilIN = Image.open(file).convert(mode="L")
    pilIN = pilIN.resize(global_size)
    imgArray = np.asarray(pilIN, dtype=np.uint8)
    temp = np.zeros(imgArray.shape, dtype=np.float64)
    temp[imgArray > threshold] = 1
    temp[temp == 0] = -1
    return temp, pilIN.size

def array2img(data, outFile=None):
    y = np.zeros(data.shape, dtype=np.uint8)
    y[data == 1] = 255
    y[data == -1] = 0
    img = Image.fromarray(y, mode="L")
    if outFile is not None:
        img.save(outFile)
    return img

path = os.listdir('image')
h = Hopfield(path)
path = os.listdir('noisy_image')
h.evaluation(path)