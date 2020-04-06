from __future__ import absolute_import, division, print_function, unicode_literals
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset
import random



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



# Step 3: Create ART classifier
classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)



# Step 4: Evaluate the ART classifier on benign test examples
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

predictions = classifier.predict(x_test)
accuracy_benign = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy_benign * 100))



# Step 5: Collect 10 instances of each case from test examples
def exract_ten_classes( data, labels, classes=(0,1,2,3,4,5,6,7,8,9), no_instance=10 ):
    x_pre = []
    y_pre = []
    for class_label in range(0, 10):
        index = random.randint(0, 500)
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



# Step 6: Generate adversarial test examples
attack = FastGradientMethod(classifier=classifier, eps=0.05)
x_test_adv = attack.generate(x=x_test_adv_pre)



# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy_adv = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_adv, axis=1)) / len(y_test_adv)
print("Accuracy on adversarial test examples: {}%".format(accuracy_adv * 100))



# Step 8: Plot Results
for ind in range(0, 100, 5):
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('Adversarial Attack On Victim Model', fontsize=24, fontweight='bold')
    columns = 2
    rows = 7
    ax = []

    ax.append(fig.add_subplot(rows, columns, 1))
    plt.text(0.38, 0.1, 'Original Image', fontsize=16, fontweight='bold')
    plt.axis('off')

    ax.append(fig.add_subplot(rows, columns, 2))
    plt.text(0.35, 0.1, 'Adversarial Image', fontsize=16, fontweight='bold')
    plt.axis('off')

    imageindex = ind
    for i in range(2, columns*rows - 2):
        if (i % 2 == 0):
            sample_pre = x_test_adv_pre[ imageindex, :]
            ax.append( fig.add_subplot(rows, columns, i + 1) )
            label_pre = np.argmax(classifier.predict(sample_pre.reshape((1, sample_pre.shape[0], sample_pre.shape[1], sample_pre.shape[2]))))
            plt.text(33, 18, 'Data:\nTrue Label = %d\nClassifier Predicted Label = %d' % (np.argmax(y_test_adv[imageindex]), label_pre))
            plt.imshow(sample_pre)
        else:
            sample_post = x_test_adv[ imageindex, :]
            ax.append( fig.add_subplot(rows, columns, i + 1) )
            label_post = np.argmax(classifier.predict(sample_post.reshape((1, sample_post.shape[0], sample_post.shape[1], sample_post.shape[2]))))
            plt.text(33, 18, 'Data:\nTrue Label = %d\nClassifier Predicted Label = %d' % (np.argmax(y_test_adv[imageindex]), label_post))
            plt.imshow(sample_post)
            imageindex = imageindex + 1

    ax.append(fig.add_subplot(rows, columns, 13))
    plt.text(0.3, 0.5, "Accuracy on benign test examples: {}%".format(accuracy_benign * 100), fontsize=10, fontweight='bold')
    plt.axis('off')

    ax.append(fig.add_subplot(rows, columns, 14))
    plt.text(0.3, 0.5, "Accuracy on benign test examples: {}%".format(accuracy_adv * 100), fontsize=10, fontweight='bold')
    plt.axis('off')

    fig.tight_layout(h_pad=4.0, w_pad=4.0)
    plt.show()