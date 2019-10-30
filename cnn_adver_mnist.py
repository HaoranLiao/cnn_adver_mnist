import tensorflow as tf
import pickle as pk
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

class Attack_Model():
    def __init__(self, model_path):
        with open('train_images.txt', 'rb') as train_im:
            self.train_images = pk.load(train_im)
        with open('train_labels.txt', 'rb') as train_la:   
            self.train_labels = pk.load(train_la)
        with open('test_images.txt', 'rb') as test_im:   
            self.test_images = pk.load(test_im)
        with open('test_labels.txt', 'rb') as test_la:   
            self.test_labels = pk.load(test_la)

        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()

        self.loss_object = tf.keras.losses.CategoricalCrossentropy()

    def create_adversarial_pattern(self):
        self.predictions = self.model.predict(self.test_images)
        with tf.GradientTape() as tape:
            for i in range(self.test_images.shape[0]):
                self.test_images = self.test_images[:, :, :, 0]
                self.test_image = self.test_images[i, :]
                self.test_image = self.test_image[None, ...]
                tape.watch(tf.convert_to_tensor(self.test_image))
                self.losses = self.loss_object(self.test_labels[i, :], self.predictions[i, :])

                self.gradients = tape.gradient(self.losses, tf.convert_to_tensor(self.test_image))
                self.signed_grads = tf.sign(self.gradients)

        return self.signed_grads

with tf.Session() as sess:
    Attack = Attack_Model('./model_1.h5')
    signed_grads = Attack.create_adversarial_pattern()