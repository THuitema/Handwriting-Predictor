# Handwriting-Predictor
**GOAL:** predict handwriting using a convolutional neural network model.

![gif of digits being predicted](images/digitPredictDemo2.gif)

## Description
This is one of my first major machine learning projects. A convolutional neural network model was developed using *Tensorflow* and *Keras* and trained with the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset to detect handwritten digits. The model I developed recorded a **98.9% accuracy** when executing `model.evaluate` for the MNIST test images. 

### Future Goals: 
- [x] create a GUI interface to predict mouse-drawn digits
- [ ] implement recognition for letters (a-z)
- [ ] implement recognition for symbols (e.g. +-/*$#)
- [ ] distinguish between spaces and characters of the same word

## Libraries Used
- **Tensorflow & Keras:** used to develop the convolutional neural network model and access the MNIST dataset
- **NumPy:** used to format/prepare data that feeds into the neural network model
- **OpenCV:** used for turning user images into a format where the characters can be detected (e.g. blurring, thresholding, and contouring)
- **MatPlotLib:** used for visualizing NumPy arrays as images and predictions from the neural network model
- **Tkinter:** used as the framework for the GUI where input is received and output is displayed


## Updates
- **March 31:** able to detect & predict digits (0-9) from an image
![picture of numbers with predictions](images/predictDigits1.png)
  
- **April 1st:** detect digits drawn by the user in a tkinter GUI (see GIF at top)

