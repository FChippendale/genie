import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import keras

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Felix Chippendale\AppData\Local\Tesseract-OCR\tesseract.exe'

%matplotlib inline

def img_resize(img, img_size):
    return cv2.resize(img, img_size, interpolation = cv2.INTER_AREA)

def read_img(img_path):
    return cv2.imread(img_path)

def create_emptyimg(img_size):
    return np.zeros([img_size[1], img_size[0]])


def extract_roi(image, img_size = (156,34), verbose = False):
    '''Function extracting the ROI and preprocessing it.
    If no ROI is detected, empty image of the specified size is returned. Note that the annotated input image is never resized.
    Args:
        param1 (str): The path of input image.
        [param2 (2-tuple): The size of output ROI.]
        [param3 (bool, [False]), specify whether the input image is returned with annotations.]
    Returns:
        bool = False (default): roi_gray, roi_thresh
        bool = True: img, roi_gray, roi_thresh
    '''
     
    (ih,iw,_) = image.shape
    img = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # reduce noise
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    blur = gray
    # edge detection using Canny
    canny = cv2.Canny(blur, 50, 150)

    contours, hierarch = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    contour = []

    # loop over the contours
    for cnt in contours:
        ## Get the stright bounding rect
        x,y,w,h = cv2.boundingRect(cnt)
        (cx,cy) = (int(x+w/2), int(y+h/2))

        if (w > 100 and h > 25) and (w < 300 and h < 100) and (iw/2-0.05*iw < cx < iw/2+0.05*iw and ih/2-0.05*ih < cy < ih/2+0.05*ih):
            contour.append(cnt)
            detected = True

            if w < 5.6 * h:
                w = 5.6 * h
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
        

        if verbose is False:
            return img_resize(roi_gray,img_size), img_resize(roi_thresh,img_size)
        else:
            # Reference
            cv2.drawMarker(img, (int(iw/2),int(ih/2)), (0,255,0),markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)
            cv2.rectangle(img, (int(iw/2-0.05*iw),int(ih/2-0.05*ih)), (int(iw/2+0.05*iw),int(ih/2+0.05*ih)), (0,255,0), 2)
            # draw contour
            cv2.drawContours(img, contour, -1, (255, 0, 0), 2)

            ## Draw rect
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2, 16)
            cv2.drawMarker(img, (cx,cy), (0,0,255),markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)

            return img, img_resize(roi_gray,img_size), img_resize(roi_thresh,img_size)


    else:

        empty_img = create_emptyimg(img_size)

        if verbose is False:
            return img_resize(empty_img,img_size), img_resize(empty_img,img_size)
        else:
            cv2.putText(img, 'No meter reading detected', (int(iw/2-200),int(ih/2-100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return img, img_resize(empty_img,img_size), img_resize(empty_img,img_size)

def remove_border(image):
    for i, col in enumerate(image[:]):
        if np.count_nonzero(col) > 25:
            image[:, i] = np.zeros(28, np.uint8)
    
    for i, row in enumerate(image):
        if np.count_nonzero(row) > 17:
            image[i] = np.zeros(28, np.uint8)
            
    return image

def find_digit(image):
    image = np.uint8(image)
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros((28, 28, 3), np.uint8)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    color = (0, 255, 0)
    for c in cnts:
        # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if 4 < cX < 24 and 4 < cY < 24:
                #cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
                cv2.drawContours(blank, [c], -1, color, -1)
                color = (0, 0, 0)
                
    return blank[:, :, 1]

def center_digit(image):
    top = 28
    bot = 0
    left = 28
    right = 0
    blank = np.zeros(image.shape, np.uint8)
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
    '''Function extracting digits from thresholded ROI. 
    These digits are then processed to produce an image of same resolution
    and characteristics of those found in the MNIST Dataset.
    Args:
        param1 (np.array): The region of interest
        [param2 (int): The expected number of digit in the image
        [param3 (bool, [True]), specify whether to discard the last digit]
    Returns:
        bool = True (default): list of digits of length num_digits - 1
        bool = False: list of digits of length num_digits
    '''
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

def open_all_images():
    '''Function creates generator for saved images, allowing for mass processing of all saved images
    Yields next picture in folder'''
    folder = './gas_photos/'
    for filename in os.listdir(folder):
        yield read_img(str(folder + filename))
        
def display_digits(digits):
    f = plt.figure()
    for i, digit in enumerate(digits):
        f.add_subplot(1, len(digits), i + 1)
        plt.imshow(digit)
        plt.axis('off')
    plt.show()

def load_model():
    """Loads model"""
    return keras.models.load_model('model.h5', custom_objects = {"softmax_v2": tf.nn.softmax})

def infer_digit(digit, model):
    prediction = model.predict(digit)
    return np.argmax(prediction)

def infer_digit_ocr(digit):
    digit_flip = np.where(digit == 0, 255, 0)
    digit_large = img_resize(np.uint8(digit_flip), (256, 256))
    text = pytesseract.image_to_string(digit_large, lang='eng', config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
    if text != '':
        return int(text)
    return 0
            

def main():
    model = load_model()
    for img in open_all_images():
        _, thresh = extract_roi(img)
        digits = extract_digits(thresh, num_digits = 8, remove_last = True)
        display_digits(digits)
        total = 0
        total_ocr = 0
        for i, digit in enumerate(digits):
            total_ocr += infer_digit_ocr(digit) * 10 ** (7 - i)
            digit = digit.reshape(1, 28, 28, 1)
            total += infer_digit(digit, model) * 10 ** (7 - i)
        #print('total: ', total)
        print('total_ocr: ', total_ocr)
        
main()
