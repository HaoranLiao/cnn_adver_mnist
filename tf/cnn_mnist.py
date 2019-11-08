import keras
from keras.datasets import mnist
import numpy as np
import skimage.transform
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
import tensorflow as tf
import pickle as pk
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

class DataGenerator:
    def __init__(self):
        mnist_data = mnist.load_data()
        self.train_images = mnist_data[0][0]
        self.train_labels = mnist_data[0][1]
        self.test_images = mnist_data[1][0]
        self.test_labels = mnist_data[1][1]

    def resize_images(self, images, shape):
        num_images = images.shape[0]
        new_images_shape = (num_images, shape[0], shape[1])
        new_images = skimage.transform.resize(
            images,
            new_images_shape,
            anti_aliasing=True,
            mode='constant')
        return new_images

    def select_digits(self, images, labels, digits):
        cumulative_test = (labels == digits[0])
        for digit in digits[1:]:
            digit_test = (labels == digit)
            cumulative_test = np.logical_or(digit_test, cumulative_test)

        valid_images = images[cumulative_test]
        valid_labels = labels[cumulative_test]
        return (valid_images, valid_labels)

class Model():
    def __init__(self):
        self.all_train_images = DataGenerator().resize_images(DataGenerator().train_images, image_size) 
        self.all_test_images = DataGenerator().resize_images(DataGenerator().test_images, image_size)
        self.all_train_labels, self.all_test_labels = DataGenerator().train_labels, DataGenerator().test_labels

        self.train_images, self.train_labels = DataGenerator().select_digits(self.all_train_images, self.all_train_labels, digits)
        self.test_images, self.test_labels = DataGenerator().select_digits(self.all_test_images, self.all_test_labels, digits)

        self.train_images, self.train_labels = self.train_images[0:sample_size], self.train_labels[0:sample_size]
        self.test_images, self.test_labels = self.test_images[0:sample_size], self.test_labels[0:sample_size]

        if keras.backend.image_data_format() == 'channels_first':
            self.train_images = self.train_images.reshape(sample_size, 1, image_size[0], image_size[1])
            self.test_images = self.test_images.reshape(sample_size, 1, image_size[0], image_size[1])
            self.input_shape = (1, image_size[0], image_size[1])
        else:
            self.train_images = self.train_images.reshape(sample_size, image_size[0], image_size[1], 1)
            self.test_images = self.test_images.reshape(sample_size, image_size[0], image_size[1], 1)
            self.input_shape = (image_size[0], image_size[1], 1)

        num_category = 10
        self.train_labels = keras.utils.to_categorical(self.train_labels, num_category)
        self.test_labels = keras.utils.to_categorical(self.test_labels, num_category)

        self.model = Sequential()

    def construct_model(self):
        self.model.add(Conv2D(32,kernel_size=2,activation='relu',input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32,kernel_size=2,activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32,kernel_size=2,strides=1,padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Conv2D(64,kernel_size=2,activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64,kernel_size=2,activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64,kernel_size=2,strides=2,padding='same',activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def save_data(self):
        with open("train_images.txt", "wb") as train_im:
            pk.dump(self.train_images, train_im)
        with open("train_labels.txt", "wb") as train_la:   
            pk.dump(self.train_labels, train_la)
        with open("test_images.txt", "wb") as test_im:   
            pk.dump(self.test_images, test_im)
        with open("test_labels.txt", "wb") as test_la:   
            pk.dump(self.test_labels, test_la)

    def train_model(self, config, train_accs):
        (batch_size, num_epoch, repeat) = config
        for i in range(repeat):
            model_log = self.model.fit(self.train_images, self.train_labels,
                      batch_size=batch_size,
                      epochs=num_epoch,
                      verbose=1,
                      validation_data=(self.train_images, self.train_labels))
            self.model.save('model_%s.h5'%repeat)
            print(model_log.history['val_acc'][-1])
            train_accs.append(model_log.history['val_acc'][-1])

        print(train_accs)


    def test_model(self, test_accs):
        for i in range(repeat):
            model = tf.keras.models.load_model('model_%s.h5'%repeat)
            score = model.evaluate(self.test_images, self.test_labels, verbose=0)
            print('Test accuracy:', score[1])
            test_accs.append(score[1])

        print(test_accs)

image_size = [8, 8]
digits = [3, 5]
sample_size = 250
data_config = (image_size, digits, sample_size)

batch_size = 100
num_epoch = 50
repeat = 1
train_config = (batch_size, num_epoch, repeat)

Model = Model()
Model.construct_model()
Model.save_data()

test_accs, train_accs = [], []
#Model.train_model(train_config, train_accs)
#print('Avg Train Accs: %.3f'%np.mean(train_accs))

#Model.test_model(test_accs)  
#print('Avg Test Accs: %.3f'%np.mean(test_accs))    



