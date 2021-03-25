import os
import matplotlib.pyplot as plt
import tensorflow as tf

num_epochs = 10
batch_size = 32
learning_rate = 0.001
data_dir = './data'
train_bottle_dir = data_dir + '/train/塑料瓶/'
train_chopsticks_dir = data_dir + '/train/一次性筷子/'
tf.enable_eager_execution()


def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)  # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label


if __name__ == '__main__':
    # 构建训练数据集
    train_bottle_filenames = tf.constant([train_bottle_dir + filename for filename in os.listdir(train_bottle_dir)])
    train_chopsticks_filenames = tf.constant(
        [train_chopsticks_dir + filename for filename in os.listdir(train_chopsticks_dir)])
    train_filenames = tf.concat([train_bottle_filenames, train_chopsticks_filenames], axis=-1)
    train_labels = tf.concat([
        tf.zeros(train_bottle_filenames.shape, dtype=tf.int32),
        tf.ones(train_chopsticks_filenames.shape, dtype=tf.int32)],
        axis=-1)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=23000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for image, label in train_dataset:
        plt.title(label.numpy())
        plt.imshow(image.numpy()[:, :, 0])
        plt.show()
        break
