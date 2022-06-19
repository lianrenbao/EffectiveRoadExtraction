import matplotlib.pyplot as plt
from PIL import Image
import os

test_img = 'e:\\image8.jpg'

if __name__ == '__main__':
    img = Image.open(test_img)
    plt.figure()
    plt.imshow(img)
    plt.show()
