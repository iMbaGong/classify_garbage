import tensorflow as tf
import numpy as np
import math, time, os


def parse_label(path):
    raw = open(path).read()
    classnames = []
    for cls in raw.split('\n'):
        if cls == '':
            continue
        classnames.append(cls)
    return classnames


def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)  # 读取原始文件
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [224, 224]) / 255.0
    return image_resized, label


def load_data(dirs, random):
    images_name = []
    labels = []
    for i, file_dir in enumerate(dirs):
        filename = [file_dir + '/' + filename for filename in os.listdir(file_dir)]
        images_name = np.concatenate([images_name, filename], axis=-1)
        labels = np.concatenate([labels, [i] * len(filename)], axis=-1)
    nums = images_name.shape[0]
    print("total:%d" % nums)
    if random:
        index = tf.random.uniform([nums], 0, nums, tf.int32)
        sample_images = images_name[index]
        sample_labels = labels[index]
        train_dataset = tf.data.Dataset.from_tensor_slices((sample_images, sample_labels))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((images_name, labels))
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset


class BaggingModel(tf.keras.Model):
    def __init__(self, base_num):
        super(BaggingModel, self).__init__()
        self.base_num = base_num
        for i in range(base_num):
            setattr(self, 'base' + str(i),
                    tf.keras.applications.MobileNetV3Small(weights='bagging/base-MobileNetV3-' + str(i) + '.h5',
                                                           classes=40))
            getattr(self, 'base' + str(i)).trainable = False

            # setattr(self, 'base' + str(i), tf.keras.models.load_model('bagging/base-MobileNetV3-' + str(i) + '.h5'))

        for i in range(40):
            setattr(self, 'dense' + str(i),
                    tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Constant(1 / 3)))

    @tf.function
    def call(self, inputs):
        X = []
        for i in range(self.base_num):
            X.append(getattr(self, 'base' + str(i))(inputs, training=False))

        outputs = []
        for i in range(40):
            outputs.append(
                getattr(self, 'dense' + str(i))(tf.concat([x[:, i:i + 1] for x in X], -1)))
        return tf.math.softmax(tf.concat(outputs, -1))


def train_base(base_num, load=False):
    num_epochs = 15
    batch_size_base = 30
    learning_rate_base = 0.001
    buffer_size_base = 7500

    train_dir = './data/train/'
    class_names = parse_label(train_dir + 'classes_label.txt')
    train_file_dirs = [train_dir + classname for classname in class_names]

    for model_no in range(base_num):
        train_dataset = load_data(train_file_dirs, random=True).shuffle(buffer_size_base).batch(batch_size_base)
        if load:
            model = tf.keras.models.load_model('bagging/base-MobileNetV3-' + str(model_no) + '.h5')
        else:
            model = tf.keras.applications.MobileNetV3Small(weights=None, classes=40)
        loss_func = tf.keras.losses.sparse_categorical_crossentropy
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_base)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    y_pred = model(images, training=True)
                    loss = tf.reduce_mean(loss_func(y_true=labels, y_pred=y_pred))
                    print("no:%d epoch:%d batch:%d loss:%f" % (model_no, epoch, i, loss))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            model.save('bagging/base-MobileNetV3-' + str(model_no) + '.h5')


def train_bagging(base_num):
    num_epochs = 3
    batch_size_bagging = 20
    learning_rate_bagging = 0.005
    buffer_size_bagging = 10000

    train_dir = './data/train/'
    class_names = parse_label(train_dir + 'classes_label.txt')
    train_file_dirs = [train_dir + classname for classname in class_names]
    model = BaggingModel(base_num)
    train_dataset = load_data(train_file_dirs, random=False).shuffle(buffer_size_bagging).batch(batch_size_bagging)
    loss_func = tf.keras.losses.sparse_categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_bagging)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(images, training=True)
                loss = tf.reduce_mean(loss_func(y_true=labels, y_pred=y_pred))
                print("Bagging epoch:%d batch:%d loss:%f" % (epoch, i, loss))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        model.save('bagging/Bagging-' + str(base_num) + '-MobileNetV3')


def evaluate():
    batch_size = 30
    buffer_size = 1200
    test_dir = './data/test/'
    class_names = parse_label(test_dir + 'classes_label.txt')
    test_file_dirs = [test_dir + classname for classname in class_names]
    test_dataset = load_data(test_file_dirs, random=False).shuffle(buffer_size).batch(batch_size)
    model = tf.keras.models.load_model('bagging/base-MobileNetV3-0.h5')
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    model.evaluate(test_dataset, batch_size=batch_size)


if __name__ == '__main__':
    base_num = 5
    train_base(base_num, load=False)
    train_bagging(base_num)
