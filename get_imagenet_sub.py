import tensorflow as tf
import os

imagenet_dir = 'G:/ILSVRC2012_img_train/'
raw = open('./data/imagenet_sub/classes_label.txt').read()
for line in raw.split('\n'):
    [file_no, class_name] = line.split(' ')
    src_dir = imagenet_dir + file_no + '.tar'
    dst_dir = './data/imagenet_sub/' + class_name + '.tar'
    if not tf.io.gfile.exists(class_name):
        print(src_dir+' is not exist')
    else:
        print(src_dir + ' copying to ' + dst_dir)
        tf.io.gfile.copy(class_name, dst_dir)



