import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#load data
data = keras.datasets.mnist

#traintest split
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ["0","1","2","3","4","5","6","7","8","9"]

#all of our images are 28x28 pixels with rgb values (256 max)
#to reduce the number/extent of computations you do the following
train_images = train_images/255.0
test_images = test_images/255.0

#the model begins here
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28))) #make list of lists into a single list
model.add(keras.layers.Dense(128, activation="relu")) #hidden layer with rectified linear unit activation
model.add(keras.layers.Dense(10, activation="softmax")) #assigns each output neuron a probability s.t the total of these probabilities is 1

#model compilation
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#fit the model, epochs is the number of times the model will see the same data
model.fit(train_images, train_labels, epochs=5)

#saving model
model.save("project.h5")


model = keras.models.load_model("project.h5")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


prediction = model.predict([image])
for i in range(len([image])): #this shows us 5 test images, alter this to see however many u want
    plt.grid(False)
    plt.imshow(image[0], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + image_label)
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])  # argmax finds the element with the largest value in the list and returns its index
    plt.show()

    predict = model.predict([image])
    print(predict[0])
    t = np.argmax(predict[0])
    print("i predict this number is ", t)









