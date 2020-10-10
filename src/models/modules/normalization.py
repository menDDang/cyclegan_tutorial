import tensorflow as tf
    

class BatchNorm2D(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, epsilon=0.001, **kwargs):
        super().__init__(self, **kwargs)

        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shapes):
        # Get shape of inputs
        _, _, _, channels = input_shapes  # [batch_size, height, width, channels]
        
        # Add non trainable variables for mean, variance
        self.mean = self.add_weight(name='mean', shape=[1, 1, 1, channels], dtype=self.dtype, trainable=False, initializer=tf.initializers.Zeros)
        self.variance = self.add_weight(name='variance', shape=[1, 1, 1, channels], dtype=self.dtype, trainable=False, initializer=tf.initializers.Ones)

        # Add trainable variables
        self.gamma = self.add_weight(name='gamma', shape=[1, 1, 1, channels], dtype=self.dtype, trainable=True)
        self.beta = self.add_weight(name='beta', shape=[1, 1, 1, channels], dtype=self.dtype, trainable=True)
        self.built = True

    def call(self, x, training=False):
        if training:
            # while training, we normalize the inputs using its mean and variance
            # Mean
            mean = tf.reduce_mean(x, axis=(0, 1, 2), keepdims=True)
            # Variance
            variance = tf.reduce_mean((x - mean) ** 2, axis=(0, 1, 2), keepdims=True)
            # Normalize inputs
            x_hat = (x - mean) * 1.0 / tf.math.sqrt(variance + self.epsilon)
            # Update self.mean & self.variance using momentum
            self.mean.assign(self.mean * self.momentum + mean * (1.0 - self.momentum))
            self.variance.assign(self.variance * self.momentum + variance * (1.0 - self.momentum))
        else:
            # while testing, we normalize the inputs uing the pre-computed mean and variance
            x_hat = (x - self.mean) * 1.0 / tf.math.sqrt(variance + self.epsilon)
        
        # scale and shift
        out = self.gamma * x_hat + self.beta
        return out


class InstanceNorm2D(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, epsilon=0.001, **kwargs):
        super().__init__(self, **kwargs)

        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shapes):
        # Get shape of inputs
        batch_size, _, _, channels = input_shapes  # [batch_size, height, width, channels]
        
        # Add non trainbale variables for mean, variance
        self.mean = self.add_weight(name='mean', shape=[batch_size, 1, 1, channels], dtype=self.dtype, trainable=False, initializer=tf.initializers.Zeros)
        self.variance = self.add_weight(name='variance', shape=[batch_size, 1, 1, channels], dtype=self.dtype, trainable=False, initializer=tf.initializers.Ones)

        # Add trainable variables
        self.gamma = self.add_weight(name='gamma', shape=[batch_size, 1, 1, channels], dtype=self.dtype, trainable=True)
        self.beta = self.add_weight(name='beta', shape=[batch_size, 1, 1, channels], dtype=self.dtype, trainable=True)
        self.built = True

    def call(self, x, training=False):
        if training:
            # while training, we normalize the inputs using its mean and variance
            # Mean
            mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
            # Variance
            variance = tf.reduce_mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
            # Normalize inputs
            x_hat = (x - mean) * 1.0 / tf.math.sqrt(variance + self.epsilon)
            # Update self.mean & self.variance using momentum
            self.mean.assign(self.mean * self.momentum + mean * (1.0 - self.momentum))
            self.variance.assign(self.variance * self.momentum + variance * (1.0 - self.momentum))
        else:
            # while testing, we normalize the inputs uing the pre-computed mean and variance
            x_hat = (x - self.mean) * 1.0 / tf.math.sqrt(variance + self.epsilon)
        
        # scale and shift
        out = self.gamma * x_hat + self.beta
        return out



    

    