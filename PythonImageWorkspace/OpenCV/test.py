import time
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from rembg.bg import remove as remove_bg
import cv2
def show_rembg(path):
    fig = plt.figure(figsize=(10, 10))

    # show original image
    fig.add_subplot(1, 2, 1)
    orig_img = Image.open(path)
    plt.imshow(orig_img)

    # show bg removed image
    fig.add_subplot(1, 2, 2)
    f = np.fromfile(path)

    started = time.time()
    result = remove_bg(f)
    elapsed = time.time() - started
    print(f'it takes {elapsed} seconds for removing bg.')

    # img = Image.open(io.BytesIO(result)).convert("RGBA")
    img = Image.frombytes('RGBA', (128,128), result, 'raw')
    plt.imshow(img)
    plt.show()
    cv2.imwrite('translated.png', img)

# Usage
show_rembg('car.jpeg')