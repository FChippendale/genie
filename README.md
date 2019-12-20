# Gas meter monitoring using OpenCV and convolutional-recurrent neural network

Finds readout by searching for a rectangle of roughly the right size

Divides readout into individual digits

Removes noise from digits and centers them

Classifies digits via CNN trained on the MNIST dataset, using Tensorflow and Keras

TODO:
-- Train network with more noise to better identify digits
-- Fine tune readout aquisition 
