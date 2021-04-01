import cv2 as cv
import numpy as np
from tensorflow import keras
import imutils
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hides tensorflow warnings


image = cv.imread('images/IMG_1127.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (9, 9), 0)
threshold, thresh = cv.threshold(blurred, 110, 255, cv.THRESH_BINARY) # pixels below 140 are turned 255 (white), above turned 0 (black)
edges = cv.Canny(thresh, 30, 150) # min val, max val

# cv.imshow('image', edges)

contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sort_contours(contours, 'left-to-right')[0]

chars = []
chars_dimensions = []
for c in contours:
    (x, y, w, h) = cv.boundingRect(c) # bounding box around contour
    # print(w, h)
    if (20 <= w) and (50 <= h):
        # extract the character and threshold it to make the character
        # appear as *white* (foreground) on a *black* background, then
        # grab the width and height of the thresholded image
        roi = gray[y:y + h, x:x + w]
        thresh = cv.threshold(roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        (threshH, threshW) = thresh.shape

        # if the width is greater than the height, resize along the
        # width dimension
        if threshW > threshH:
            thresh = imutils.resize(thresh, width=28)
        # otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=28)

        (threshH, threshW) = thresh.shape
        # dX = int(max(0, 28 - threshW) / 2.0)
        # dY = int(max(0, 28 - threshH) / 2.0)
        # adding borders to make chars smaller in image, more similar to MNIST images
        dX = 8
        dY = 8

        padded = cv.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv.BORDER_CONSTANT, value=(0, 0, 0))
        padded = cv.resize(padded, (28, 28)) # resizing to 28x28, same as MNIST dataset
        # padded = padded.reshape((padded[0].shape, 28, 28, 1))


        # prepare the padded image for classification
        padded = padded.astype("float32") / 255.0 # normalizing
        padded = np.expand_dims(padded, axis=-1)

        # update our list of characters that will be predicted
        chars.append(padded) # [padded, (x, y, w, h), None]
        chars_dimensions.append((x, y, w, h))

# PREDICTING
chars = np.asarray(chars)
# print(type([chars]), chars.shape, [chars])
model = keras.models.load_model('models/acc-0.989')
predictions = model.predict([chars])

for count, p in enumerate(predictions):
    print(p) # predictions for all possible nums
    number = np.argmax(p)  # predicted num (rounded to whole)
    rounded = '{0:.4f}'.format(p[number]) # rounded = str(round(p[number], 4))
    x, y, w, h = chars_dimensions[count]
    cv.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), thickness=2) # rectangle around char
    cv.putText(image, f'{number} ({rounded})', (x, y-12), cv.FONT_HERSHEY_SIMPLEX, 1.0, (36, 255, 12), thickness=2) # cv.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0),
    # plt.imshow(chars[count]) # predicted image

    # plt.title(f'Prediction: {number} (' + str(rounded) + '%)') # rounded doesn't round in the f-string for some reason
    # plt.show()
    print('\n')


cv.imshow('predicted', image) # with boxes around detected chars & their predictions

cv.waitKey(0)
