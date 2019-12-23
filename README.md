# Gas meter monitoring using OpenCV and convolutional-recurrent neural network

Finds readout by searching for a rectangle of roughly the right size
Uses fact that surrounding area is all white to improved acquisition

Divides readout into individual digits

Removes noise from digits and centers them

Classifies digits via tesseract ocr package and CNN trained on the MNIST dataset, using Tensorflow and Keras

TODO:
- Retrain network with more variation to better identify digits
- Add debug mode

