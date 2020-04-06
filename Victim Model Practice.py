import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from keras.datasets import cifar10
from art.utils import load_dataset

classifier_url ="https://tfhub.dev/deepmind/ganeval-cifar10-convnet/1" #@param {type:"string"}
IMAGE_SHAPE = (32, 32)

classifier = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))])
classifier.summary()

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10")) # Original Dataset
print("x_train shape: " + str(x_train.shape) + "\n" + "x_train size: " + str(x_train.size) + "\n" +
      "y_train shape: " + str(y_train.shape) + "\n" + "y_train size: " + str(y_train.size) + "\n" +
      "x_test shape: " + str(x_test.shape) + "\n" + "x_test size: " + str(x_test.size) + "\n" +
      "y_test shape: " + str(y_test.shape) + "\n" + "y_test size: " + str(y_test.size) + "\n")

predictions = classifier.predict(x_test)
accuracy_benign = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy_benign * 100))
































"""
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
from keras.datasets import cifar10

classifier_url ="https://tfhub.dev/deepmind/ganeval-cifar10-convnet/1" #@param {type:"string"}
IMAGE_SHAPE = (32, 32)

classifier = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))])
classifier.summary()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
sample = x_test[ 4, : ]
plt.imshow(sample)
plt.show()

sample = np.array(sample)/255.0
print(sample.shape)

result = classifier.predict(sample[np.newaxis, ...])
print(result.shape)

predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

"""