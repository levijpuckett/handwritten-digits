# MNIST_NN
An implementation of an artificial neural network (ANN). Trained using gradient descent and back propagation, the ANN attempts to identify handwritten digits from the MNIST database.

*This code is based on the free online book "Neural Networks and Deep Learning," by Michael Nielsen. It is an excellent introduction to the concept of neural networks, and includes an implementation. Available:* http://neuralnetworksanddeeplearning.com


## MNIST Dataset
The MNIST dataset is a well known machine learning dataset. Over 60,000 images of handwritten digits from the NIST database have been centered and size-normalized, making for an excellent training set. All digits are labelled, and a subset of 10,000 are withheld for out-of-sample testing. An example of an MNIST digit can be seen below.

![alt text][Screen Shot 2019-01-17 at 5.23.47 PM.png]

## Neural Network
The neural network structure consists of nodes, each with many inputs and outputs. Each input has a weight, and each node is associated with a bias. The network is an object realized as 2 arrays: a 2D array of weights, and a 1D array of biases. A graphical interpretation of the NN is used to get a visual sense of what is happening when it is being trained. As the weights and biases are updated, the colour of the nodes and branches vary accordingly.

![alt text][Screen Shot 2019-01-17 at 5.23.47 PM.png]

The NN uses a simple sigmoid function for its logistic function. It is trained using backpropagation and gradient descent over multiple epochs. In each epoch, the entire training dataset is shown to the NN, and then the training set is shuffled and the next epoch begins. 10,000 images are withheld for a measure of generalization error.
