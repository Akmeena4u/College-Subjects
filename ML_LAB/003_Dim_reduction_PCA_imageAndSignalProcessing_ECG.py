'''What is the role of Principal Component Analysis (PCA) as an effective method for reducing dimensionality, and how does it find applications across diverse domains including image processing, signal processing,
and data compression. Utilize PCA in the context of both image processing and signal processing, particularly with regard to EEG data for reducing dimensionality.

Note: Please use the provided EEG dataset and the previously provided image for dimensionality reduction.

sol-
Principal Component Analysis (PCA) is a powerful technique for dimensionality reduction, which plays a crucial role in various domains, including image processing, signal processing, and data compression.
It helps in reducing the number of features (dimensions) while preserving as much information as possible. 

Image Processing with PCA:
input- image  represented as a two-dimensional array, where each element rep intensity , for grascale it lies into 0-255 single val and in colored in 
Output:  an image, but with a reduced number of dimensions or features compared to the original image. This reduced image will have fewer pixels or channels.
purpose -By reducing the dimensionality, you can simplify the image representation, making it computationally more efficient for various tasks like image compression, feature extraction, or pattern recognition
'''
#import libraries
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image = Image.open('your_image.png')
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert the image to grayscale and flatten it
image_gray = np.array(image.convert('L'))
image_flat = image_gray.reshape(-1, 1)


# Initialize PCA with the desired number of components
n_components = 50
pca = PCA(n_components=n_components)
# Fit PCA on the flattened image
pca.fit(image_flat)
# Transform the image data to a lower-dimensional space
image_reduced = pca.transform(image_flat)
# Inverse transform to get the reduced image back to the original space
image_restored = pca.inverse_transform(image_reduced)
image_restored = image_restored.reshape(image_gray.shape)

#Display the reduced image
plt.figure(figsize=(6, 6))
plt.imshow(image_restored, cmap='gray')
plt.title(f'Reduced Image (n_components = {n_components})')
plt.axis('off')
plt.show()


#APP-2 PCA on image without using library functions----------------------------------------------------
'''Steps for PCA:
Standardize the data: Ensure that the data is centered and scaled to have a mean of 0 and a standard deviation of 1.
Compute the covariance matrix: Calculate the covariance matrix of the standardized data.
Calculate eigenvalues and eigenvectors: Find the eigenvalues and eigenvectors of the covariance matrix.
Sort eigenvalues: Arrange the eigenvalues in descending order and corresponding eigenvectors accordingly.
Select the top k eigenvectors: Choose the first k eigenvectors that represent the most variance in the data.
Project the data: Multiply the data by the selected eigenvectors to obtain the reduced-dimensional representation.'''



#---------------------------------------------PCA ON ECG DATASET -SIGNAL PROCESSING--------------------------------------------------------------------------

#impoer libraries
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#load data
# Specify the path to your EEG data file (e.g., CSV)
eeg_data_file = 'your_eeg_data.csv'
# Load EEG data using pandas (assuming CSV format)
eeg_data = pd.read_csv(eeg_data_file)

#apply pca
n_components = 10  # Choose the desired number of components
pca = PCA(n_components=n_components)
eeg_reduced = pca.fit_transform(eeg_data)

# Visualize the reduced EEG data
plt.figure(figsize=(10, 5))
plt.plot(eeg_reduced)
plt.title(f'Reduced EEG Data ({n_components} components)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


