import tensorflow as tf


class BaggingModel(tf.keras.Model):
    def __init__(self, base_num):
        super(BaggingModel, self).__init__()
        self.base_num = base_num
        for i in range(base_num):
            setattr(self, 'base' + str(i),
                    tf.keras.applications.MobileNetV3Small(weights='bagging/base-MobileNetV3-' + str(i) + '.h5',
                                                           classes=40))
            # setattr(self, 'base' + str(i), tf.keras.models.load_model('bagging/base-MobileNetV3-' + str(i) + '.h5'))

        for i in range(40):
            setattr(self, 'dense' + str(i), tf.keras.layers.Dense(1))

    def call(self, inputs):
        X = []
        for i in range(self.base_num):
            X.append(getattr(self, 'base' + str(i))(inputs))

        outputs = []
        for i in range(40):
            outputs.append(
                getattr(self, 'dense' + str(i))(tf.concat([x[:, i:i + 1] for x in X], -1)))
        return tf.math.softmax(tf.concat(outputs, -1))


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


model = MyModel()
model(tf.random.uniform([3, 40, 40, 3]))
model.save('modeltest/subclass_model_test')
