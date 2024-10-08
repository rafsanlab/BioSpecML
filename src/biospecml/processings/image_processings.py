
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

def img_inverse(img:np.ndarray, point:tuple=(0,0), background:int=0):
    """
    Inverse image using cv.bitwise_not by checking the point given whether it fit
    the background. If it's not, it inverse will be run. (this function is use to invert
    K-means 1 and 0 cluster output results).

    Args:
    - img (np.ndarray): the K-means image projection (2D image)
    - point (tuple): background coordinate
    - background (int): background value (i.e; 1 or 0)

    Returns:
    - img (np.ndarray): inversed or not inversed image
    """
    if img[point[0], point[1]] != background:
        img = cv.bitwise_not(img)
        return img
    else:
        return img


def img_thres_otsu(img, blur_kernel:tuple=(3,3), tval:int=0, maxval:int=255):
    """
    Applies Otsu thresholding to an image. Will convert 3-channels image to 2.

    Args:
    - img (numpy.ndarray): 2D or 3D image array representing the input image.
    - blur_kernel (tuple): Gaussian blur kernel size.
    - tval (int): Thresholding value.
    - maxval (int): Value to be assigned to thresholded pixels.

    Returns:
    - thresh (np.ndarray): Thresholded image array.
    """
    img, blur_kernel, tval, maxval = img, blur_kernel, tval, maxval

    # checking img input
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img = img
    else:
        raise Exception('*img input has invalid shape.')

    # img -> blur -> Otsu
    if blur_kernel is not None:
        img = cv.GaussianBlur(img, blur_kernel, 0)
    thresh = cv.threshold(img, tval, maxval, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # thresh = cv.threshold(img, tval, maxval, cv.THRESH_OTSU)[1]

    return thresh


def img_rm_debris(img, n:float=0.01):
    """
    Remove small particles from a 2D image.

    Args:
    - img (numpy.ndarray): Array representing the input 2D image.
    - n (float): Multiplier of the average area of the image. 
        - smaller n values remove larger particles.

    Returns:
    - thresh (np.ndarray): Thresholded image array.
    """

    thresh, n = img, n

    # checking img input
    if len(img.shape) != 2:
        raise Exception('*img input has invalid shape.')

    # determine average area
    average_area = [] 
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        area = w * h
        average_area.append(area)
    average = sum(average_area) / len(average_area)

    # remove 'debris'
    cnts = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < average * n:
             cv.drawContours(thresh, [c], -1, (0,0,0), -1)

    return thresh


def img_rm_holes(img, n:float=0.1, holes_kernel:tuple=(5,5), iterations:int=2):
    """
    Remove holes from an 2D image array.

    Args:
    - img (np.ndarray): an array of 2D image.
    - n (float): multiplier of average area size.
    - holes_kernel (tup): size of holes to be remove.
    - interations (int): number of iterations .

    Returns:
    - close (np.ndarray): image array.
    """
    thresh, n, iterations = img, n, iterations

    # checking img input
    if len(img.shape) != 2:
        raise Exception('*img input has invalid shape.')

    # determine average area
    average_area = [] 
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        area = w * h
        average_area.append(area)
    average = sum(average_area) / len(average_area)

    # remove 'holes'
    cnts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < average * n:
            cv.drawContours(thresh, [c], -1, (0,0,0), -1)
  
    # Morph close and invert image
    kernel = cv.getStructuringElement(cv.MORPH_RECT, holes_kernel)
    close = cv.morphologyEx(
        thresh,cv.MORPH_CLOSE,
        kernel, iterations=iterations
         )
  
    return close


def invert_mask(arr, target_value:int|float=0, coor_to_check:tuple= (0,0),
                show_mask:bool=False):
    if len(arr.shape)!=2:
        raise TypeError('Array should be 2 dimensional')
    coor_value = arr[coor_to_check[0],coor_to_check[1]]
    if coor_value != target_value:
        arr = 1 - arr
    else:
        arr = arr
    if show_mask:
        plt.imshow(arr, cmap='gray')
        plt.show()
    return arr


def apply_thresholding(img, blur_kernel=(3,3), rm_debris=True, debris_n=0.01,
                       rm_holes=True, holes_n=0.01, holes_iter=1, invert_output=False
                       ):
    """
    Apply otsu then series of processings including debris removal, holes remover and
    option to invert the fina image.

    """
    thresh = img_thres_otsu(img, blur_kernel=blur_kernel)
    if rm_debris:
        thresh = img_rm_debris(thresh, n=debris_n)
    if rm_holes:
        thresh = img_rm_holes(thresh, n=holes_n, iterations=holes_iter)
    if invert_output:
        thresh = cv.bitwise_not(thresh)
    return thresh


def extract_patches(image:Image, patch_size:tuple, stride:int):
    """
    Extracts patches of a given size from an image with a specified stride.

    Parameters:
    - image (PIL.Image): The input image.
    - patch_size (tuple): The (width, height) of each patch.
    - stride (int): The stride to move across the image.

    Returns:
    - List of patches (PIL.Image).

    Raises:
    - ValueError: If the image size does not allow for perfect patch division.
    """

    img_width, img_height = image.size
    patch_width, patch_height = patch_size

    # # Check if patches fit perfectly within the image
    # if (img_width - patch_width) % stride != 0 or (img_height - patch_height) % stride != 0:
    #     raise ValueError("The patches size and stride do not fit perfectly into the image dimensions.")

    patches, idx = [], []
    i = 0
    # Loop to extract patches
    for top in range(0, img_height - patch_height + 1, stride):
        for left in range(0, img_width - patch_width + 1, stride):
            box = (left, top, left + patch_width, top + patch_height)
            patch = image.crop(box)  # Crop the patch from the image
            patches.append(patch)
            idx.append(i)
            i += 1

    return patches, idx


def center_crop_to_size(image, preferred_size:tuple):
    """
    Center crop an image to the prefrred size.
        - image: must be a PIL.Image object.

    """

    if not isinstance(image, Image.Image):
        raise TypeError("The input image must be a PIL Image object.")

    img_width, img_height = image.size
    crop_width, crop_height = preferred_size

    # Ensure the preferred size does not exceed the image size
    crop_width = min(crop_width, img_width)
    crop_height = min(crop_height, img_height)

    # Calculate the cropping box
    left = (img_width - crop_width) // 2
    top = (img_height - crop_height) // 2
    right = (img_width + crop_width) // 2
    bottom = (img_height + crop_height) // 2

    # Perform the crop
    return image.crop((left, top, right, bottom))