import tensorflow as tf


class ResnetGenerator(tf.keras.Model):
    def __init__(self, ngf, output_nc, nblocks):
        super(ResnetGenerator, self).__init__()
        self.nblocks = nblocks

        self.downsampling = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=ngf, kernel_size=7, padding='same', activation=None),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=ngf * 2, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=ngf * 4, kernel_size=3, strides=2, padding='same', activation='relu')
        ])

        self.residual_blocks = [tf.keras.layers.Conv2D(filters=ngf * 4, kernel_size=3, padding='same', activation='relu') for i in range(nblocks)]

        self.upsampling = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=ngf*2, kernel_size=3, strides=2, padding='same', output_padding=1, activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=ngf, kernel_size=3, strides=2, padding='same', output_padding=1, activation='relu')
        ])

        self.conv = tf.keras.layers.Conv2D(output_nc, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, x):
        # down sampling layer
        x = self.downsampling(x)

        # residual convolution layer
        for i in range(self.nblocks):
            x = self.residual_blocks[i](x) + x

        # up-sampling layer
        x = self.upsampling(x)

        # output layer
        x = self.conv(x)

        return x

