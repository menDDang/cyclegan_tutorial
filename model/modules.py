import tensorflow as tf
import tensorflow_addons as tfa


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


class NLayerDiscriminator(tf.keras.Model):
    def __init__(self, nlayers, ndf):
        super(NLayerDiscriminator, self).__init__()
        self.nlayers = nlayers

        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=ndf, kernel_size=4, strides=2, padding='valid'),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])

        # gradually increase the number of filters
        sequence = []
        for n in range(1, nlayers):
            sequence.append(tf.keras.layers.Conv2D(filters=ndf * min(2**n, 8), kernel_size=4, strides=2, padding='valid'))
            sequence.append(tfa.layers.InstanceNormalization())
            sequence.append(tf.keras.layers.LeakyReLU(alpha=0.2))
        sequence.append(tf.keras.layers.Conv2D(ndf * min(2**nlayers, 8), kernel_size=4, strides=1, padding='valid'))
        sequence.append(tfa.layers.InstanceNormalization())
        sequence.append(tf.keras.layers.LeakyReLU(alpha=0.2))

        self.conv2 = tf.keras.Sequential(sequence)

        self.conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='valid')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


if __name__ == "__main__":
    import numpy as np

    nc = 3
    ngf = 256
    nblocks = 3

    A = np.zeros(shape=[1, 100, 100, nc], dtype=np.float32)
    generator = ResnetGenerator(ngf, nc, nblocks)
    discriminator = NLayerDiscriminator()

    A2B = generator(A)
    print(A2B.shape)

    p = discriminator(A2B)
    print(p.shape)
