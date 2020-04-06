from __future__ import absolute_import, division, print_function, unicode_literals
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset


# Step 1: Load the CIFAR 10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10")) # Original Dataset
print("x_train shape: " + str(x_train.shape) + "\n" + "x_train size: " + str(x_train.size) + "\n" +
      "y_train shape: " + str(y_train.shape) + "\n" + "y_train size: " + str(y_train.size) + "\n" +
      "x_test shape: " + str(x_test.shape) + "\n" + "x_test size: " + str(x_test.size) + "\n" +
      "y_test shape: " + str(y_test.shape) + "\n" + "y_test size: " + str(y_test.size) + "\n")

    # Sampled dataset to train model
x_train, y_train = x_train[:5000], y_train[:5000] # take 5000 samples for the training set
x_test, y_test = x_test[:1000], y_test[:1000] # take 1000 samples for the testing set
print("x_train shape: " + str(x_train.shape) + "\n" + "x_train size: " + str(x_train.size) + "\n" +
      "y_train shape: " + str(y_train.shape) + "\n" + "y_train size: " + str(y_train.size) + "\n" +
      "x_test shape: " + str(x_test.shape) + "\n" + "x_test size: " + str(x_test.size) + "\n" +
      "y_test shape: " + str(y_test.shape) + "\n" + "y_test size: " + str(y_test.size) + "\n")



# Step 2: Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



# Step 3: Create the classifier
classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)



# Step 4: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))



# Step 5: Collect 10 instances of each case from test examples
def exract_ten_classes( data, labels, classes=(0,1,2,3,4,5,6,7,8,9), no_instance=10 ):
    x_pre = []
    y_pre = []
    for class_label in range(0, 10):
        index = 0
        iteration = no_instance
        while (iteration != 0):
            if np.argmax(labels[index]) == classes[class_label]:
                x_pre.append(data[index])
                y_pre.append(int(class_label))
                iteration = iteration - 1
            index = index + 1
    x = np.asarray(x_pre)
    y = np.asarray(y_pre)
    return x, y

x_test_adv_pre, y_test_adv = exract_ten_classes( x_test, y_test )
y_test_adv = keras.utils.to_categorical( y_test_adv, 10 )
print("x_test_adv_pre shape: " + str(x_test_adv_pre.shape) + "\n" + "x_test_adv_pre size: " + str(x_test_adv_pre.size) + "\n" +
      "y_test_adv_pre shape: " + str(y_test_adv.shape) + "\n" + "y_test_adv_pre size: " + str(y_test_adv.size) + "\n")

# picking a test sample before generating adversarial examples
sample_pre = x_test_adv_pre[ 1, :]
print( sample_pre.shape )
plt.imshow( sample_pre )
plt.axis( 'off' )
plt.show( )
print("Label = " + str(np.argmax(y_test_adv[1])))
label_pre = np.argmax(classifier.predict( sample_pre.reshape( (1, sample_pre.shape[ 0 ], sample_pre.shape[ 1 ], sample_pre.shape[ 2 ]) ) ) )
print( 'class prediction for the test sample_pre:', label_pre )



# Step 6: Generate adversarial test examples
attack = FastGradientMethod(classifier=classifier, eps=0.25)
x_test_adv = attack.generate(x=x_test_adv_pre)

# picking a test sample after generating adversarial examples
sample_post = x_test_adv[ 1, :]
print( sample_post.shape )
plt.imshow( sample_post )
plt.axis( 'off' )
plt.show( )
print("Label = " + str(np.argmax(y_test_adv[1])))
label_post = np.argmax(classifier.predict( sample_post.reshape( (1, sample_post.shape[ 0 ], sample_post.shape[ 1 ], sample_post.shape[ 2 ]) ) ) )
print( 'class prediction for the test sample_post:', label_post )



# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_adv, axis=1)) / len(y_test_adv)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))