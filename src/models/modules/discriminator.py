import tensorflow as tf
from .normalization import InstanceNorm2D

class NLayerDiscriminator(tf.keras.Model):
    def __init__(self, num_layers, num_filters, leaky_relu_alpha=0.2, instance_momentum=0.99, **kwargs):
        super().__init__(self, **kwargs)
        
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.leaky_relu_alpha = leaky_relu_alpha
        self.instance_momentum = instance_momentum

        # Add layers
        self.conv = tf.keras.Sequential()
        self.conv.add(tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=4, strides=2, padding='valid'))
        self.conv.add(tf.keras.layers.LeakyReLU(alpha=self.leaky_relu_alpha))
        # Gradually increase the number of filters
        for n in range(1, self.num_layers):
            self.conv.add(tf.keras.layers.Conv2D(filters=self.num_filters * min(2**n, 8), kernel_size=4, strides=2, padding='valid'))
            self.conv.add(InstanceNorm2D(momentum=self.instance_momentum, dtype=self.dtype))
            self.conv.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv.add(tf.keras.layers.Conv2D(self.num_filters * min(2**self.num_layers, 8), kernel_size=4, strides=1, padding='valid'))
        self.conv.add(InstanceNorm2D(momentum=self.instance_momentum, dtype=self.dtype))
        self.conv.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        # Final convolutional layer
        self.conv.add(tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='valid'))

    def call(self, x):
        x = self.conv(x)
        return x
