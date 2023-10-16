'''Design a Neural Network Architecture to capture essential informationfrom an image by projecting it into a lower-dimensional space and subsequently reconstructing it using decoder layers. This process
involves adding noise to the image data, and the auto encoder is trained to effectively remove this noise. Calculate SNR, PSNR, SSIM and RMSE between the original and reconstructed images.

Note: Use the MNIST digit data set to perform above task.

Autoencoder--An autoencoder is a type of ANN that captures essential information from images, projects them into a lower-dimensional space, and reconstructs them with added noise
It's primarily used for dimensionality reduction, feature learning, and data compression. Autoencoders consist of two main components: an encoder and a decoder.

Encoder--It takes the original input data and maps it to a lower-dimensional representation (latent space or bottleneck). This mapping is usually done through a series of neural network layers, such as fully connected (dense) layers, convolutional layers, or recurrent layers.
Decoder:The decoder takes the reduced-dimensional representation from the encoder and attempts to reconstruct the original input data from it.

Input and Output:
input:-data for that we wants compact lower dimentional representation
output:- typically a reconstruction of the same data that encoder gives as output

Training--The autoencoder's performance is measured by how well it can reconstruct the input data.
The noisy image serves as the input, and the autoencoder is trained to output a denoised version of the same image. The key is that the encoder component of the autoencoder learns to capture essential information in a lower-dimensional space, which is then used by the decoder to produce the clean, reconstructed image.

Flow of Autoencoder
Noisy Image -> Encoder -> Compressed Representation -> Decoder -> Reconstruct Clear Image

'''

#Import Modules
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
     
#Load the Dataset
(x_train, _), (x_test, _) = mnist.load_data()


# normalize the image data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
     
# reshape in the input data for the model
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)
x_test.shape


#Add Noise to the Image
# add noise
noise_factor = 0.6
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
     
# clip the values in the range of 0-1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
     

#Exploratory Data Analysis
# randomly select input image
index = np.random.randint(len(x_test))
# plot the image
plt.imshow(x_test[index].reshape(28,28))
plt.gray()

# randomly select input image
index = np.random.randint(len(x_test))
# plot the image
plt.imshow(x_test_noisy[index].reshape(28,28))
plt.gray()


#Model Creation
model = Sequential([
                    # encoder network
                    Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
                    MaxPooling2D(2, padding='same'),
                    Conv2D(16, 3, activation='relu', padding='same'),
                    MaxPooling2D(2, padding='same'),
                    # decoder network
                    Conv2D(16, 3, activation='relu', padding='same'),
                    UpSampling2D(2),
                    Conv2D(32, 3, activation='relu', padding='same'),
                    UpSampling2D(2),
                    # output layer
                    Conv2D(1, 3, activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()


# train the model
model.fit(x_train_noisy, x_train, epochs=20, batch_size=256, validation_data=(x_test_noisy, x_test))



#Visualize the Results
# predict the results from model (get compressed images)
pred = model.predict(x_test_noisy)
     
# randomly select input image
index = np.random.randint(len(x_test))
# plot the image
plt.imshow(x_test_noisy[index].reshape(28,28))
plt.gray()

# visualize denoised image
plt.imshow(pred[index].reshape(28,28))
plt.gray()
