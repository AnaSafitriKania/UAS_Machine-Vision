import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from skimage.feature import hog
from skimage import exposure

# Load the MNIST dataset
mnist = datasets.fetch_openml('mnist_784')
data = mnist.data.astype("uint8")
target = mnist.target.astype("uint8")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Feature extraction using HOG
def extract_hog_features(images):
    features = []
    for image in images:
        fd, hog_image = hog(image.reshape((28, 28)), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
        features.append(fd)
    return np.array(features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Standardize the data
scaler = StandardScaler().fit(X_train_hog)
X_train_hog = scaler.transform(X_train_hog)
X_test_hog = scaler.transform(X_test_hog)

# Train the SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train_hog, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_hog)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

# Display results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:")
print(conf_matrix)


