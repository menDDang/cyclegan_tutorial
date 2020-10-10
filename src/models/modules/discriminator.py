import tensorflow as tf


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
