import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import gzip
import random
def load_mnist(filename, type, n_datapoints):
    
    image_size = 28
    f = gzip.open(filename)
    
    if(type == 'image'):
        f.read(16)   
        buf = f.read(n_datapoints * image_size * image_size)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(n_datapoints, image_size, image_size, 1)
    elif(type == 'label'):
        f.read(8) 
        buf = f.read(n_datapoints)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        data = data.reshape(n_datapoints, 1)
    return data
train_size = 60000
test_size = 10000


X = load_mnist('train_imagecsv.gz','image', train_size)
y = load_mnist('train_labelscsv.gz', 'label', train_size)
X_test = load_mnist('test_imagecsv.gz', 'image', test_size)
y_test = load_mnist('test_labelscsv.gz', 'label', test_size)
index = random.randint(0, train_size)
print('Index: ', index)
print('Training Set: ')
print('Label:', y[index])
img = np.asarray(X[index]).squeeze()
plt.imshow(img)
plt.show()

index = random.randint(0, test_size)
print('Index: ', index)
print('Training Set: ')
print('Label:', y_test[index])
img = np.asarray(X_test[index]).squeeze()
plt.imshow(img)
plt.show()

t1 = X_test[index].reshape(1, 28*28)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X[:(train_size//10)], y[:(train_size//10)], test_size=0.25, random_state=28)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

score = []
k=1
while k<12:
    print('Begin KNN with k=',k)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier = classifier.fit(X_train.reshape(X_train.shape[0], 28*28), y_train)
    pred = classifier.predict(X_valid.reshape(X_valid.shape[0], 28*28))
    accuracy = accuracy_score(y_valid, pred)
    score.append(accuracy)
    print("Accuracy: ",accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_valid, pred))
    print("\n---------------\n")
    k=k+2
    
print('Training the Model')
classifier = KNeighborsClassifier(n_neighbors=5)
classifier = classifier.fit(X.reshape(X.shape[0], 28*28), y)

print('Testing the Model')
y_pred = classifier.predict(X_test.reshape(X_test.shape[0], 28*28))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_pred, y_test))

import joblib

joblib.dump(classifier, 'knn_model.gzip', compress=('gzip',3))