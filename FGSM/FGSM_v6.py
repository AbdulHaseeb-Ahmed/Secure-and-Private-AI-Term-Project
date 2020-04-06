from __future__ import absolute_import, division, print_function, unicode_literals
import keras
import numpy as np
import matplotlib.pyplot as plt
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset
import random

import tensorflow as tf
import tensorflow_hub as hub
tf.compat.v1.disable_eager_execution()



# Step 1: Load the CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10")) # Original Dataset
print("x_train shape: " + str(x_train.shape) + "\n" + "x_train size: " + str(x_train.size) + "\n" +
      "y_train shape: " + str(y_train.shape) + "\n" + "y_train size: " + str(y_train.size) + "\n" +
      "x_test shape: " + str(x_test.shape) + "\n" + "x_test size: " + str(x_test.size) + "\n" +
      "y_test shape: " + str(y_test.shape) + "\n" + "y_test size: " + str(y_test.size) + "\n")
print()



# Step 2: Load the victim model
classifier_url ="https://tfhub.dev/deepmind/ganeval-cifar10-convnet/1" #@param {type:"string"}
IMAGE_SHAPE = (32, 32)
classifier = KerasClassifier(model=tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))]), clip_values=(min_, max_))



# Step 3: Evaluate the victim model on the benign dataset
predictions = classifier.predict(x_test)
accuracy_benign = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%\n".format(accuracy_benign * 100))



# Step 4: Collect 10 instances of each case from test examples
def exract_ten_classes( data, labels, classes=(0,1,2,3,4,5,6,7,8,9), no_instance=10 ):
    x_pre = []
    y_pre = []
    for class_label in range(0, 10):
        index = random.randint(0, 1000)
        iteration = no_instance
        while (iteration != 0):
            if np.argmax(labels[index]) == classes[class_label]:
                x_pre.append(data[index])
                y_pre.append(int(class_label))
                iteration = iteration - 1
            index = index + 1
    x = np.asarray(x_pre)
    y = keras.utils.to_categorical(np.asarray(y_pre), 10)
    return x, y

x_test_adv_pre, y_test_adv = exract_ten_classes( x_test, y_test )



# Step 5: Generate adversarial test examples
attack = FastGradientMethod(classifier=classifier, eps=0.5)
x_test_adv = attack.generate(x=x_test_adv_pre)



# Step 6: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy_adv = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_adv, axis=1)) / len(y_test_adv)
print("Accuracy on adversarial test examples: {}%".format(accuracy_adv * 100))



# Step 7: Plot Results
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

