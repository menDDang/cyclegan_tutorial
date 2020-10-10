import os
import cv2
import numpy as np
import tensorflow as tf

def load_img(filename):
    img = cv2.imread(filename,flags=cv2.IMREAD_COLOR)
    return img


class DataSet:
    def __init__(self, hp, train=True):
        self.hp = hp
        self.train = train

        if train:
            self.dir_A = os.path.join(hp.data.dir, "trainA")
            self.dir_B = os.path.join(hp.data.dir, "trainB")
        else:
            self.dir_A = os.path.join(hp.data.dir, "testA")
            self.dir_B = os.path.join(hp.data.dir, "testB")

        self.list_A = os.listdir(self.dir_A)
        self.list_B = os.listdir(self.dir_B)

    def gen(self):
        for i in range(len(self.list_A)):
            if self.train:
                j = np.random.randint(0, len(self.list_B))
            else:
                j = i

            # get file names
            img_A_name = os.path.join(self.dir_A, self.list_A[i])
            img_B_name = os.path.join(self.dir_B, self.list_B[j])

            # read images
            img_A = cv2.imread(img_A_name, flags=cv2.IMREAD_COLOR).astype(np.float32)
            img_B = cv2.imread(img_B_name, flags=cv2.IMREAD_COLOR).astype(np.float32)

            # normalize images
            img_A /= 255.0
            img_B /= 255.0

            yield img_A, img_B


def create_data_loader(hp, train=True):
    dataset = DataSet(hp, train=train)
    generator = dataset.gen
    A_shape = [hp.data.size, hp.data.size, hp.data.nc_A]
    B_shape = [hp.data.size, hp.data.size, hp.data.nc_B]
    dataloader = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape(A_shape), tf.TensorShape(B_shape))
    )

    return dataloader.repeat()
