from model.modules import *
import tensorflow as tf


class CycleGAN(tf.keras.Model):
    def __init__(self, hp):
        super(CycleGAN, self).__init__()
        self.hp = hp

        # define lambda values
        self.lambda_A = hp.model.lambda_A
        self.lambda_B = hp.model.lambda_B
        self.lambda_idt = hp.model.lambda_idt

        # define generators
        ngf = hp.model.ngf; output_nc = hp.model.nc_B; nblocks = hp.model.n_res_block
        self.G_A = ResnetGenerator(ngf, output_nc, nblocks)  # convert image from A to B
        nfg = hp.model.ngf; output_nc = hp.model.nc_A; nblocks = hp.model.n_res_block
        self.G_B = ResnetGenerator(ngf, output_nc, nblocks)  # convert image from B to A

        # define discriminators
        nlayers = hp.model.n_D_layers; ndf = 64
        self.D_A = NLayerDiscriminator(nlayers, ndf)
        self.D_B = NLayerDiscriminator(nlayers, ndf)

        # define optimizers
        self.optimizer_G = tf.optimizers.Adam()
        self.optimizer_D = tf.optimizers.Adam()

    def generate(self, inputs, mode='A2B'):
        if not mode in ['A2B', 'B2A']:
            raise ValueError('mode is either "A2B" or "B2A"!')

        if mode == 'A2B':
            return self.G_A(inputs)

        else:
            return self.G_B(inputs)

    def adversarial_loss(self, prediction, target_is_real):
        # Create label
        if target_is_real:
            label = tf.ones_like(prediction)
        else:
            label = tf.zeros_like(prediction)

        # Binary Cross Entropy loss
        loss = tf.keras.losses.binary_crossentropy(label, prediction, from_logits=True)
        return loss

    def L1_norm_loss(self, real, generated):
        return tf.reduce_mean(tf.abs(real - generated))

    def train_discriminator(self, D, fake, real):
        with tf.GradientTape as tape:
            # Discriminate real samples
            pred_real = D(real)
            loss_D_real = self.adversarial_loss(pred_real, True)

            # Discriminate fake samples
            pred_fake = D(fake)
            loss_D_fake = self.adversarial_loss(pred_fake, False)

            loss = (loss_D_real + loss_D_fake) / 2
        grads = tape.gradient(loss, D.trainable_varialbe)
        self.optimizer_D.apply_gradients(zip(grads, D.trainable_varialbe))
        return loss

    def train_on_batch(self, batch_A, batch_B):
        # Generate samples
        with tf.GradientTape as tape:
            fake_B = self.G_A(batch_A)  # A --> B
            fake_A = self.G_B(batch_B)  # B --> A

        # Train discriminators
        loss_D_A = self.train_discriminator(self.D_A, fake_A, batch_A)
        loss_D_B = self.train_discriminator(self.D_B, fake_B, batch_B)
        loss_D = (loss_D_A + loss_D_B) / 2

        # Train Generators
        with tf.GradientTape as tape:
            # Identity loss
            # G_A should be identity if real_B is fed : || G_A(B) - B ||
            idt_A = self.G_A(batch_B)
            loss_idt_A = self.L1_norm_loss(idt_A, batch_B) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed : || G_B(A) - A ||
            idt_B = self.G_B(batch_A)
            loss_idt_B = self.L1_norm_loss(idt_B, batch_A) * self.lambda_A * self.lambda_idt

            #
            loss_G_A = self.adversarial_loss(self.D_A(fake_B), True)
            loss_G_B = self.adversarial_loss(self.D_B(fake_A), True)

            # Cycle loss - forward
            rec_A = self.G_A(fake_A)  # B --> A --> B
            rec_B = self.G_B(fake_B)  # A --> B --> A
            loss_cycle_A = self.L1_norm_loss(rec_A, batch_A) * self.lambda_A
            loss_cycle_B = self.L1_norm_loss(rec_B, batch_B) * self.lambda_B

            # Combine loss & calculate gradients
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            grads = tape.gradient(loss_G, [self.G_A.trainable_variables, self.G_B.trainable_variables])
            self.optimizer_G.apply_gradients(zip(grads, [self.G_A.trainable_variables, self.G_B.trainable_variables]))

        return loss_D, loss_G
