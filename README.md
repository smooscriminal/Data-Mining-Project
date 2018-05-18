# Data-Mining-Project

DL.py
CNN models can be trained with fair accuracy despite of limited number of training data. 
Thus, 2500 from both classes of food and drinks are first filtered out from the entire dataset, 
excluding the irrelevant image files and saved under a training directory. 
Smiliarily, 2000 food and drink images are chosen and saved under a testing directory.
Images are then imported into the image generating function in the Keras library which the function will rescale, cast rotations,
and flips to generate even more images to prevent overfitting problem.
Three layers of the convolutional neural networks are added as the hidden layers with relu as the activation function. 
The weight are saved and added to the next level in order to save processing power.
The output of numbers such as the loss rate, accuracy, and percentage done are reported on the console by the Keras library.

Add the path to the directory to use the code
