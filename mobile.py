import os
import matplotlib.pyplot as plt
import tensorflow as tf


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


def load_data(dirs):
    images_name = []
    labels = []
    for i, file_dir in enumerate(dirs):
        filename = tf.constant([file_dir + '/' + filename for filename in os.listdir(file_dir)])
        images_name = tf.concat([images_name, filename], axis=-1)
        labels = tf.concat([labels, tf.constant(i, shape=filename.shape[0])], axis=-1)
    print("total:%d" % images_name.shape[0])
    train_dataset = tf.data.Dataset.from_tensor_slices((images_name, labels))
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return train_dataset


num_epochs = 5
batch_size = 30
learning_rate_dense = 0.1
learning_rate_mobile = 0.001
buffer_size = 10000

train_dir = './data/imagenet_sub/'
class_names = parse_label(train_dir + 'classes_label.txt')
train_file_dirs = [train_dir + classname for classname in class_names]
train_dataset = load_data(train_file_dirs)
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# model = tf.keras.applications.MobileNetV2(weights=None, classes=len(class_names))
model = tf.keras.models.load_model("mobilenet-garbage.h5")
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_mobile)


# @tf.function
def train_one_step(X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        tf.print("loss", loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))


for e in range(num_epochs):
    for i, (X, y) in enumerate(train_dataset):
        print("epoch:%d batch:%d" % (e, i))
        train_one_step(X, y)
    model.save('mobilenet-garbage.h5')
