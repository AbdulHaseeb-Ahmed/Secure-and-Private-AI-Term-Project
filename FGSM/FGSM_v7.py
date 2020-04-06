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
#print("Accuracy on benign test examples: {}%\n".format(accuracy_benign * 100))



# Step 4: Collect 10 instances of each case from test examples
def exract_ten_classes( data, labels, classes=(0,1,2,3,4,5,6,7,8,9), no_instance=10 ):
    x_pre = []
    y_pre = []
    for class_label in range(0, 10):
        index = random.randint(0, 5000)
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


# Step 5: Generate adversarial test examples and Evaluate the ART classifier on adversarial test examples
attack_eps_5 = FastGradientMethod(classifier=classifier, eps=0.05)
x_test_adv_eps_5 = attack_eps_5.generate(x=x_test_adv_pre)
predictions_eps_5 = classifier.predict(x_test_adv_eps_5)
accuracy_adv_eps_5 = np.sum(np.argmax(predictions_eps_5, axis=1) == np.argmax(y_test_adv, axis=1)) / len(y_test_adv)
#print("Accuracy on adversarial test examples with eps = 0.05: {}%".format(accuracy_adv_eps_5 * 100))

attack_eps_10 = FastGradientMethod(classifier=classifier, eps=0.1)
x_test_adv_eps_10 = attack_eps_10.generate(x=x_test_adv_pre)
predictions_eps_10 = classifier.predict(x_test_adv_eps_10)
accuracy_adv_eps_10 = np.sum(np.argmax(predictions_eps_10, axis=1) == np.argmax(y_test_adv, axis=1)) / len(y_test_adv)
#print("Accuracy on adversarial test examples with eps = 0.1: {}%".format(accuracy_adv_eps_10 * 100))

attack_eps_50 = FastGradientMethod(classifier=classifier, eps=0.5)
x_test_adv_eps_50 = attack_eps_50.generate(x=x_test_adv_pre)
predictions_eps_50 = classifier.predict(x_test_adv_eps_50)
accuracy_adv_eps_50 = np.sum(np.argmax(predictions_eps_50, axis=1) == np.argmax(y_test_adv, axis=1)) / len(y_test_adv)
#print("Accuracy on adversarial test examples with eps = 0.5: {}%".format(accuracy_adv_eps_50 * 100))

attack_eps_95 = FastGradientMethod(classifier=classifier, eps=0.95)
x_test_adv_eps_95 = attack_eps_95.generate(x=x_test_adv_pre)
predictions_eps_95 = classifier.predict(x_test_adv_eps_95)
accuracy_adv_eps_95 = np.sum(np.argmax(predictions_eps_95, axis=1) == np.argmax(y_test_adv, axis=1)) / len(y_test_adv)
#print("Accuracy on adversarial test examples with eps = 0.95: {}%".format(accuracy_adv_eps_95 * 100))

accuracies = [accuracy_adv_eps_5 * 100, accuracy_adv_eps_10 * 100, accuracy_adv_eps_50 * 100, accuracy_adv_eps_95 * 100]


# Step 6: Plot Results
for ind in range(0, 100, 5):
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('Adversarial Attack On Victim Model', fontsize=24, fontweight='bold')
    columns = 5
    rows = 7
    ax = []

    ax.append(fig.add_subplot(rows, columns, 1))
    plt.text(0.38, 0.1, 'Original Image', fontsize=10, fontweight='bold')
    plt.axis('off')

    eps = [0.05, 0.1, 0.5, 0.95]
    for i in range(2, 6):
        ax.append(fig.add_subplot(rows, columns, i))
        plt.text(0.0, 0.1, 'Adversarial Image EPS = ' + str(eps[i - 2]), fontsize=10, fontweight='bold')
        plt.axis('off')

    imageindex = ind
    for i in range(5, columns*rows - 6, 5):
        sample_pre = x_test_adv_pre[ imageindex, :]
        ax.append( fig.add_subplot(rows, columns, i + 1) )
        label_pre = np.argmax(classifier.predict(sample_pre.reshape((1, sample_pre.shape[0], sample_pre.shape[1], sample_pre.shape[2]))))
        plt.text(33, 18, 'Data:\nTrue Label = %d\nPredicted Label = %d' % (np.argmax(y_test_adv[imageindex]), label_pre))
        plt.imshow(sample_pre)

        sample_post_eps_5 = x_test_adv_eps_5[ imageindex, :]
        ax.append( fig.add_subplot(rows, columns, i + 2) )
        label_post_eps_5 = np.argmax(classifier.predict(sample_post_eps_5.reshape((1, sample_post_eps_5.shape[0], sample_post_eps_5.shape[1], sample_post_eps_5.shape[2]))))
        plt.text(33, 18, 'Data:\nTrue Label = %d\nPredicted Label = %d' % (np.argmax(y_test_adv[imageindex]), label_post_eps_5))
        plt.imshow(sample_post_eps_5)

        sample_post_eps_10 = x_test_adv_eps_10[imageindex, :]
        ax.append(fig.add_subplot(rows, columns, i + 3))
        label_post_eps_10 = np.argmax(classifier.predict(sample_post_eps_10.reshape((1, sample_post_eps_10.shape[0], sample_post_eps_10.shape[1], sample_post_eps_10.shape[2]))))
        plt.text(33, 18, 'Data:\nTrue Label = %d\nPredicted Label = %d' % (np.argmax(y_test_adv[imageindex]), label_post_eps_10))
        plt.imshow(sample_post_eps_10)

        sample_post_eps_50 = x_test_adv_eps_50[imageindex, :]
        ax.append(fig.add_subplot(rows, columns, i + 4))
        label_post_eps_50 = np.argmax(classifier.predict(sample_post_eps_50.reshape((1, sample_post_eps_50.shape[0], sample_post_eps_50.shape[1], sample_post_eps_50.shape[2]))))
        plt.text(33, 18, 'Data:\nTrue Label = %d\nPredicted Label = %d' % (np.argmax(y_test_adv[imageindex]), label_post_eps_50))
        plt.imshow(sample_post_eps_50)

        sample_post_eps_95 = x_test_adv_eps_95[imageindex, :]
        ax.append(fig.add_subplot(rows, columns, i + 5))
        label_post_eps_95 = np.argmax(classifier.predict(sample_post_eps_95.reshape((1, sample_post_eps_95.shape[0], sample_post_eps_95.shape[1], sample_post_eps_95.shape[2]))))
        plt.text(33, 18, 'Data:\nTrue Label = %d\nPredicted Label = %d' % (np.argmax(y_test_adv[imageindex]), label_post_eps_95))
        plt.imshow(sample_post_eps_95)

        imageindex = imageindex + 1


    ax.append(fig.add_subplot(rows, columns, 31))
    plt.text(0.0, 0.5, "Accuracy on benign test examples: {}%".format(round(accuracy_benign * 100),2), fontsize=8, fontweight='bold')
    plt.axis('off')

    for i in range(32, 36):
        ax.append(fig.add_subplot(rows, columns, i))
        plt.text(0.0, 0.5, "Accuracy on test examples eps = " + str(eps[i-32]) + ": {}%".format(round(accuracies[i-32]), 2), fontsize=8, fontweight='bold')
        plt.axis('off')

    fig.tight_layout(h_pad=4.0, w_pad=4.0)
    plt.show()


# Step 7: Data from Results
print()
print("Accuracy on benign test examples: {}%".format(accuracy_benign * 100))
print("Accuracy on adversarial test examples with eps = 0.05: {}%".format(accuracies[0]))
print("Accuracy on adversarial test examples with eps = 0.1: {}%".format(accuracies[1]))
print("Accuracy on adversarial test examples with eps = 0.5: {}%".format(accuracies[2]))
print("Accuracy on adversarial test examples with eps = 0.95: {}%".format(accuracies[3]))
print()

all_count = []
for j in range(0, 100, 10):
    count = [0, 0, 0, 0, 0]
    for i in range(j, j + 10):
        sample_pre = x_test_adv_pre[ i, : ]
        label_pre = np.argmax(classifier.predict(sample_pre.reshape((1, sample_pre.shape[0], sample_pre.shape[1], sample_pre.shape[2]))))
        sample_post_eps_5 = x_test_adv_eps_5[ i, : ]
        label_post_eps_5 = np.argmax(classifier.predict(sample_post_eps_5.reshape((1, sample_post_eps_5.shape[0], sample_post_eps_5.shape[1], sample_post_eps_5.shape[2]))))
        sample_post_eps_10 = x_test_adv_eps_10[ i, : ]
        label_post_eps_10 = np.argmax(classifier.predict(sample_post_eps_10.reshape((1, sample_post_eps_10.shape[0], sample_post_eps_10.shape[1], sample_post_eps_10.shape[2]))))
        sample_post_eps_50 = x_test_adv_eps_50[ i, : ]
        label_post_eps_50 = np.argmax(classifier.predict(sample_post_eps_50.reshape((1, sample_post_eps_50.shape[0], sample_post_eps_50.shape[1], sample_post_eps_50.shape[2]))))
        sample_post_eps_95 = x_test_adv_eps_95[ i, : ]
        label_post_eps_95 = np.argmax(classifier.predict(sample_post_eps_95.reshape((1, sample_post_eps_95.shape[0], sample_post_eps_95.shape[1], sample_post_eps_95.shape[2]))))
        if (label_pre == np.argmax(y_test_adv[i])):
            count[0] = count[0] + 1
        if (label_post_eps_5 == np.argmax(y_test_adv[i])):
            count[1] = count[1] + 1
        if (label_post_eps_10 == np.argmax(y_test_adv[i])):
            count[2] = count[2] + 1
        if (label_post_eps_50 == np.argmax(y_test_adv[i])):
            count[3] = count[3] + 1
        if (label_post_eps_95 == np.argmax(y_test_adv[i])):
            count[4] = count[4] + 1
    all_count.append(count)
print()

Labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
for i in range(0, 10):
    print("Classifier with benign example has " + str(Labels[i]) + " recognition accuracy of = " + str((all_count[i][0] / 10) * 100) + "%")
    print("Fast Gradient Method with eps = 0.05 has " + str(Labels[i]) + " recognition accuracy of = " + str((all_count[i][1] / 10) * 100) + "%")
    print("Fast Gradient Method with eps = 0.10 has " + str(Labels[i]) + " recognition accuracy of = " + str((all_count[i][2] / 10) * 100) + "%")
    print("Fast Gradient Method with eps = 0.50 has " + str(Labels[i]) + " recognition accuracy of = " + str((all_count[i][3] / 10) * 100) + "%")
    print("Fast Gradient Method with eps = 0.95 has " + str(Labels[i]) + " recognition accuracy of = " + str((all_count[i][4] / 10) * 100) + "%")
    print()