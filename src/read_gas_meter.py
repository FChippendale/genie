import os
import numpy as np
import cv2
import tensorflow as tf
import keras
import pytesseract

class InputModeError(Exception):
    """Raised when mode is not any of the supported inputs, supported inputs are:
        - image_direct
        - image_saved
        - all_image_from_folder"""
    pass

class InputSourceError(Exception):
    """Raised when attempting to open file which does not exists"""
    pass

class InputModelPathError(Exception):
    """Raised when model cannot be located"""
    pass

def read_img(img_path):
    """Opens image from path provided. Path must contain full image path
        - example: .//gas_photos//2019-12-10-1410.png"""
    if os.path.isfile(img_path):
        return cv2.imread(img_path)
    else:
        raise InputSourceError(str('attempted to open file ' + img_path + ' which does not exist'))

def open_all_images(folder):
    """Function creates generator for saved images, allowing for mass processing of all saved images
    Yields next name of picture and picture in folder"""
    for filename in os.listdir(folder):
        yield filename, read_img(str(folder + filename))

def create_empty_img(img_size):
    """Function creates empty np.arrays of the specified dimensions to act as blank image
    Args:
        param1 (int, int): (x, y) dimensions of blank image
    Returns:
        np.array[int][int] of zeros
    """
    return np.zeros([img_size[1], img_size[0]], dtype=np.uint8)

def img_resize(img, img_size):
    """Function creates empty np.arrays of the specified dimensions to act as blank image
    Args:
        param1 (np.array): image to resize
        param2 (int, int): (x, y) dimensions of output image
    Returns:
        (np.array): image of requested dimensions
    """
    return cv2.resize(img, img_size, interpolation = cv2.INTER_AREA)

def show_image(img, name = ''):
    """Opens a new window to display images if debug mode is active, these windows can be closed by
    pressing the [0] key
    Args:
        param1 (np.array): image to show
        param2 (str, ['']): title to give window opened containing image
    """
    cv2.imshow(name,img)
    cv2.waitKey(0)
    return

def extract_roi(image, img_size = (156,34)):
    """Function extracting the ROI and preprocessing it.
    If no ROI is detected, empty image of the specified size is returned.
    Note that the annotated input image is never resized.
    Args:
        param1 (str): The path of input image.
        param2 (2-tuple): The size of output ROI.
    Returns:
        roi_gray, roi_thresh
    """
    
    ih, iw, _ = image.shape
    img = image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_thresh = np.uint8(np.where(gray > 140, 255, 0))
    kernel = np.ones((5,5),np.uint8)
    gray_thresh = cv2.erode(gray_thresh,kernel,iterations = 2)
    gray_thresh = cv2.dilate(gray_thresh,kernel,iterations = 2)
    
    
    canny = cv2.Canny(gray_thresh, 50, 150)
    contours, hierarch = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    contour = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        (cx,cy) = (int(x+w/2), int(y+h/2))

        if (w > 100 and h > 25) and (w < 300 and h < 100) and (iw/2-0.05*iw < cx < iw/2+0.05*iw and ih/2-0.05*ih < cy < ih/2+0.05*ih):
            contour.append(cnt)
            detected = True
            break

    if detected is True:
        h = 37
        w = h * 7
        img_roi = image[y:y+h, x:int(x+w)]
        roi_gray = cv2.bitwise_not(cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY))
        roi_blur = cv2.medianBlur(roi_gray,3)

        roi_thresh = cv2.adaptiveThreshold(roi_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,2)

        kernel = np.ones((2,2),np.uint8)
        roi_thresh = cv2.erode(roi_thresh,kernel,iterations = 2)
        roi_thresh = cv2.dilate(roi_thresh,kernel,iterations = 2)
        
        return img_resize(roi_gray,img_size), img_resize(roi_thresh,img_size)
    else:
        empty_img = create_empty_img(img_size)
        return img_resize(empty_img,img_size), img_resize(empty_img,img_size)
        

def remove_border(image):
    """Removes solid borders around digit to reduce noise
    Args: 
        param1 (np.array): input image to process
    Returns:
        (np.array): image of same size as input where solid edges around digit have been eliminated
    """
    for i, col in enumerate(image[:]):
        if np.count_nonzero(col) > 25:
            image[:, i] = np.zeros(28, np.uint8)
    
    for i, row in enumerate(image):
        if np.count_nonzero(row) > 17:
            image[i] = np.zeros(28, np.uint8)
            
    return image

def find_digit(image):
    """eliminates all but the largest contour near the center, to remove noise.
    Input image must be 28x28 resolution.
    Params:
        param1 (np.array[28, 28]): image to locate digit in
    Returns:
        (np.array[28, 28]): output image containing only the digit of size 28x28
    """
    image = np.uint8(image)
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros((28, 28, 3), np.uint8)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    color = (0, 255, 0)
    
    for c in cnts:
        M = cv2.moments(c)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if 4 < cX < 24 and 4 < cY < 24:
                cv2.drawContours(blank, [c], -1, color, -1)
                color = (0, 0, 0)
                
    return blank[:, :, 1]

def center_digit(image):
    """Centers digit in image. Input image must be 28x28 resolution.
    Params:
        param1 (np.array[28, 28]): image to center digit in
    Returns:
        (np.array[28, 28]): output image containing centered digit of size 28x28
    """
    top = 28
    bot = 0
    left = 28
    right = 0
    blank = create_empty_img(image.shape)
    for i in range(14):
        if np.count_nonzero(image[i]) != 0:
            top = min(top, i)
        if np.count_nonzero(image[27 - i]) != 0:
            bot = max(bot, 27 - i)
        if np.count_nonzero(image[:, i]) != 0:
            left = min(left, i)
        if np.count_nonzero(image[:, 27 - i]) != 0:
            right = max(right, 27 - i)
    clip = image[top:bot + 1, left:right + 1]
    blank[(image.shape[0] // 2) - (clip.shape[0] // 2):(image.shape[0] // 2) - (clip.shape[0] // 2) + clip.shape[0], (image.shape[1] // 2) - (clip.shape[1] // 2):(image.shape[1] // 2) - (clip.shape[1] // 2) + clip.shape[1]] = clip
            
    return blank
            

def extract_digits(image, num_digits = 8, remove_last = True):
    """Function extracting digits from thresholded ROI. 
    These digits are then processed to produce an image of same resolution
    and characteristics of those found in the MNIST Dataset.
    Args:
        param1 (np.array): The region of interest
        param2 (int): The expected number of digit in the image
        param3 (bool, [True]), specify whether to discard the last digit
    Returns:
        bool = True (default): list of digits of length num_digits - 1
        bool = False: list of digits of length num_digits
    """
    digits = [image[:, i:i+image.shape[1] // num_digits] for i in range(0, image.shape[1], image.shape[1] // num_digits)]
    if digits[-1].shape != digits[0].shape:
        digits = digits[:-1:]
    if remove_last:
        digits = digits[:-1:]
        
    digits_resized = [img_resize(digit, (28, 28)) for digit in digits]
    digits_thresh = [np.where(digit > 70, 1, 0) for digit in digits_resized]
    digits_clipped = [remove_border(digit) for digit in digits_thresh]
    digits_clean = [find_digit(digit) for digit in digits_clipped]
    digits_centered = [center_digit(digit) for digit in digits_clean]
    
    return digits_centered


def load_model(path = 'MNISTmodel.h5'):
    """Loads tensorflow model from given path. Model must be of .h5 format.
    Args:
        param1 (str): path to model.
    Returns:
        loaded model
    """
    if os.path.isfile(path):
        return keras.models.load_model(path, custom_objects = {"softmax_v2": tf.nn.softmax})
    else:
        raise InputModelPathError('could not find model to load')

def infer_digit_MNIST(digit, model):
    prediction = model.predict(digit)
    return np.argmax(prediction)

def prep_tesseract_digits(digits):
    digits_large = [img_resize(np.uint8(digit), (256, 256)) for digit in digits]
    kernel = np.ones((5, 5))
    digits_thin = [cv2.erode(digit, kernel, iterations = 1) for digit in digits_large]
    digits_blur = [cv2.blur(digit, (5, 5)) for digit in digits_thin]
    digits_flip = [np.where(digit == 0, 255, 0) for digit in digits_blur]
    return digits_flip
    
def infer_digit_tesseract(digit):
    text = pytesseract.image_to_string(digit, lang='eng', config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
    if text != '':
        return int(text)
    return -1
            

def read_gas_meter(mode, source, model_path, debug = False):
    r"""When given images of the gas meter, returns the readings from them.
    WARNING: filepath for tesseract.exe must be entered manually into code,
    usual path is: C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe

    Args:
        param1 (str): determines the type of source, options are [image_direct], [image_saved], [all_image_from_folder]
        param2: the source on the image, depends on param1:
                image_direct: expects (np.array) of single image
                image_saved: expects (str) with path to particular image
                all_image_from_folder: expects (str) with path to folder containing all images to analyse
        param3: path for CNN keras model trained on MNIST saved as .h5
        param4 (bool, [False]): determines whether to show the various masks and images generated
                                thorughout the extraction for debugging
    Returns:
        list (str, int) containing name and value read from each image processed
    """
        
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\FChippendale\AppData\Local\Tesseract-OCR\tesseract.exe'
    
    if mode == 'image_direct':
        input_images = [('NA', np.uint8(source))]
    elif mode == 'image_saved':
        input_images = [(source, read_img(source))]
    elif mode == 'all_image_from_folder':
        input_images = open_all_images(source)
    else:
        raise InputModeError("mode parameter was not of supported type, use help(InputModeError) for details")        
    
    model = load_model(model_path)
    predictions = []
    
    for path, img in input_images:
        _, thresh = extract_roi(img)
        digits = extract_digits(thresh, num_digits = 8, remove_last = False)
        digits_ocr = prep_tesseract_digits(digits)
        total = 0
        total_ocr = 0
        
        for i, digit, digit_ocr in zip(range(len(digits)), digits, digits_ocr):
            val = infer_digit_tesseract(digit_ocr)
            if val == -1:
                digit = digit.reshape(1, 28, 28, 1)
                if np.count_nonzero(digit) != 0:
                    val = infer_digit_MNIST(digit, model)
                else:
                    val = 0
                total += val * 10 ** (7 - i)
            else:
                total_ocr += val * 10 ** (7 - i)
            
        predictions.append((path, total+total_ocr))
    
    return predictions
        
