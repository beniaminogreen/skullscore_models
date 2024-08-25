import numpy as np
from PIL import Image
from PIL import Image

import torch
import torchvision.transforms as transforms

# code from [here](https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil)
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

# code from [here](https://stackoverflow.com/questions/53131926/i-want-to-create-salt-and-pepper-noise-function-pil-and-numpy)
def add_salt_and_pepper(image, prob=0.05):
    # If the specified `prob` is negative or zero, we don't need to do anything.
    if prob <= 0:
        return image

    arr = np.asarray(image)
    original_dtype = arr.dtype

    # Derive the number of intensity levels from the array datatype.
    intensity_levels = 2 ** (arr[0, 0].nbytes * 8)

    min_intensity = 0
    max_intensity = intensity_levels - 1

    # Generate an array with the same shape as the image's:
    # Each entry will have:
    # 1 with probability: 1 - prob
    # 0 or np.nan (50% each) with probability: prob
    random_image_arr = np.random.choice(
        [min_intensity, 1, np.nan], p=[prob / 2, 1 - prob, prob / 2], size=arr.shape
    )

    # This results in an image array with the following properties:
    # - With probability 1 - prob: the pixel KEEPS ITS VALUE (it was multiplied by 1)
    # - With probability prob/2: the pixel has value zero (it was multiplied by 0)
    # - With probability prob/2: the pixel has value np.nan (it was multiplied by np.nan)
    # We need to to `arr.astype(np.float)` to make sure np.nan is a valid value.
    salt_and_peppered_arr = arr.astype(np.float64) * random_image_arr

    # Since we want SALT instead of NaN, we replace it.
    # We cast the array back to its original dtype so we can pass it to PIL.
    salt_and_peppered_arr = np.nan_to_num(
        salt_and_peppered_arr, nan=max_intensity
    ).astype(original_dtype)

    return Image.fromarray(salt_and_peppered_arr)


def add_gauss(image, sigma = 15):

    arr = np.asarray(image)
    original_dtype = arr.dtype
    row,col,ch= arr.shape

    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = np.clip(image + gauss, 0, 255)



    return Image.fromarray(noisy.astype(original_dtype))

def random_crop(image):
    transform = transforms.RandomCrop((int(image.size[1]*.75), int(image.size[0]*.75)))

    image_crop = transform(image)

    return(image_crop)
