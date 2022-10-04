# Hand-Gesture-Recognition

Sign Language to Text Conversion


Abstract
Sign language is one of the oldest and most natural form of language for communication, but since most people do not know sign language and interpreters are very difficult to come by we have come up with a real time method using neural networks for fingerspelling based american sign language.

In this method, the hand is first passed through a filter and after the filter is applied the hand is passed through a classifier which predicts the class of the hand gestures. This method provides 98.00 % accuracy for the 26 letters of the alphabet.

Project Description
American sign language is a predominant sign language Since the only disability D&M people have is communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language.

Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior and visuals.

Deaf and Mute(Dumb)(D&M) people make use of their hands to express different gestures to express their ideas with other people.

Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language.

Sign language is a visual language and consists of 3 major components

components

In this project I basically focus on producing a model which can recognize Fingerspelling based hand gestures in order to form a complete word by combining each gesture.
.


# Steps of building this project
1. The first Step of building this project was of creating the folders for storing the training and testing data. As, in this project I have built my own dataset.
2. The second step, after the folder creation is of creating the training and testing dataset.
I captured each frame shown by the webcam of our machine.

After capturing the image from the ROI, I applied gaussian blur filter to the image which helps for extracting various features of the image.

3. After the creation of the training and testing data. The third step is of creating a model for training. Here, I have used Convolutional Neural Network(CNN) for building this model. The model summary is as following
Convolutional Neural Network(CNN)
Unlike regular Neural Networks, in the layers of CNN, the neurons are arranged in 3 dimensions: width, height, depth.

The neurons in a layer will only be connected to a small region of the layer (window size) before it, instead of all of the neurons in a fully-connected manner.

Moreover, the final output layer would have dimensions(number of classes), because by the end of the CNN architecture we will reduce the full image into a single vector of class scores.

# 1. Convolutional Layer:
In convolution layer I have taken a small window size [typically of length 5*5] that extends to the depth of the input matrix.

The layer consists of learnable filters of window size. During every iteration I slid the window by stride size [typically 1], and compute the dot product of filter entries and input values at a given position.

As I continue this process well create a 2-Dimensional activation matrix that gives the response of that matrix at every spatial position.

That is, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some colour.

# 2. Pooling Layer:
We use pooling layer to decrease the size of activation matrix and ultimately reduce the learnable parameters.

There are two types of pooling:

# a. Max Pooling:
In max pooling we take a window size [for example window of size 2*2], and only taken the maximum of 4 values.

Well lid this window and continue this process, so well finally get an activation matrix half of its original Size.

# b. Average Pooling:
In average pooling we take average of all Values in a window.
# 3. Fully Connected Layer:
In convolution layer neurons are connected only to a local region, while in a fully connected region, well connect the all the inputs to neurons.
# 4. Final Output Layer:
After getting values from fully connected layer, well connect them to final layer of neurons [having count equal to total number of classes], that will predict the probability of each image to be in different classes.

# 4: The final step after the model has been trained is of creating a GUI that will be used to convert Sings into text and form sentence, which would be helpful for communicating with D&M people.

# Training:
I have converted our input images (RGB) into grayscale and applied gaussian blur to remove unnecessary noise. I then applied adaptive threshold to extract hand from the background and resize the images to 128 x 128.

I feed the input images after preprocessing to the model for training and testing after applying all the operations mentioned above.

The prediction layer estimates how likely the image will fall under one of the classes. So, the output is normalized between 0 and 1 and such that the sum of each value in each class sums to 1. I have achieved this using SoftMax function.

At first the output of the prediction layer will be somewhat far from the actual value. To make it better I have trained the networks using labeled data. The cross-entropy is a performance measurement used in the classification. It is a continuous function which is positive at values which is not same as labeled value and is zero exactly when it is equal to the labeled value.

Therefore, I optimized the cross-entropy by minimizing it as close to zero. To do this in my network layer I adjusted the weights of my neural network. TensorFlow has an inbuilt function to calculate the cross entropy.

As I have out the cross-entropy function, then I optimized it using Gradient Descent in fact with the best gradient descent optimizer is called Adam Optimizer.

# Testing:
While testing the applications I found out that some of the symbol predictions were coming out wrong.

So, I used two layers of algorithms to verify and predict symbols which are more similar to each other so that I can get close as I can to detect the symbol shown.

In my testing the following symbols were not showing properly and were giving output as other symbols :

For D : R and U
For U : D and R
For I : T, D, K and I
For S : M and N
So, to handle above cases I made three different classifiers for classifying these sets:

{D, R, U}
{T, K, D, I}
{S, M, N}

# 5. Results:
I have achieved an accuracy of 95.8% in my model using only layer 1 of the algorithm, and using the combination of layer 1 and layer 2 I achieve an accuracy of 98.0%.


