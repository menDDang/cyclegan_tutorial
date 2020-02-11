from model.modules import *
import tensorflow as tf
import numpy as np

class CycleGAN(tf.keras.Model):
    def __init__(self, hp):
        super(CycleGAN, self).__init__()
        self.hp = hp

        # define lambda values
        self.lambda_A = hp.model.lambda_A
        self.lambda_B = hp.model.lambda_B
        self.lambda_idt = hp.model.lambda_idt

        # define generators
        ngf = hp.model.ngf; output_nc = hp.data.nc_B; nblocks = hp.model.n_res_blocks
        self.G_A = ResnetGenerator(ngf, output_nc, nblocks)  # convert image from A to B
        ngf = hp.model.ngf; output_nc = hp.data.nc_A; nblocks = hp.model.n_res_blocks
        self.G_B = ResnetGenerator(ngf, output_nc, nblocks)  # convert image from B to A

        # define discriminators
        nlayers = hp.model.n_D_layers; ndf = 64
        self.D_A = NLayerDiscriminator(nlayers, ndf)
        self.D_B = NLayerDiscriminator(nlayers, ndf)

        # define optimizers
        lr_G = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=hp.train.start_learning_rate_G,
            decay_steps=50,
            end_learning_rate=hp.train.end_learning_rate_G
        )
        lr_D = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=hp.train.start_learning_rate_D,
            decay_steps=50,
            end_learning_rate=hp.train.end_learning_rate_D
        )
        self.optimizer_G = tf.optimizers.Adam(learning_rate=lr_G)
        self.optimizer_D = tf.optimizers.Adam(learning_rate=lr_D)

        # define loss functions
        self.loss_fn_D = tf.losses.BinaryCrossentropy(from_logits=True)

    def generate(self, inputs, mode='A2B'):
        if not mode in ['A2B', 'B2A']:
            raise ValueError('mode is either "A2B" or "B2A"!')

        if mode == 'A2B':
            return self.G_A(inputs)

        else:
            return self.G_B(inputs)

    def adversarial_loss(self, prediction, target_is_real):
        # flatten prediction
        #prediction = tf.reshape(prediction, [-1])

        # Create label
        if target_is_real:
            label = tf.ones_like(prediction)
        else:
            label = tf.zeros_like(prediction)

        # Binary Cross Entropy loss
        loss = self.loss_fn_D(y_true=label, y_pred=prediction)

        return loss

    def L1_norm_loss(self, real, generated):
        return tf.reduce_mean(tf.abs(real - generated))

    def train_discriminator(self, batch_A, batch_B, mode='A2B'):
        if mode == 'A2B':
            A = batch_A
            B = batch_B
            G = self.G_A
            D = self.D_B
        else:
            A = batch_B
            B = batch_B
            G = self.G_B
            D = self.D_A

        with tf.GradientTape() as tape:
            # Generate samples
            fake = G(A)

            # Discriminate real samples
            pred_real = D(B)
            loss_D_real = self.adversarial_loss(pred_real, True)

            # Discriminate fake samples
            pred_fake = D(fake)
            loss_D_fake = self.adversarial_loss(pred_fake, False)

            loss = (loss_D_real + loss_D_fake) / 2
            grads = tape.gradient(loss, D.trainable_variables)
            self.optimizer_D.apply_gradients(zip(grads, D.trainable_variables))

        return tf.reduce_mean(loss)

    def train_generator(self, batch_A, batch_B):
        with tf.GradientTape() as tape:
            # Generate samples
            fake_B = self.G_A(batch_A)  # A --> B
            fake_A = self.G_B(batch_B)  # B --> A
            idt_A = self.G_A(batch_B)  # B --> B
            idt_B = self.G_B(batch_A)  # B --> B
            rec_A = self.G_B(self.G_A(batch_A))  # A --> B --> A
            rec_B = self.G_A(self.G_B(batch_B))  # B --> A --> B

            # Adversarial loss
            loss_adv_A = self.adversarial_loss(self.D_B(fake_B), True)
            loss_adv_B = self.adversarial_loss(self.D_A(fake_A), True)

            # Identity loss
            loss_idt_A = self.L1_norm_loss(idt_A, batch_B) * self.lambda_A * self.lambda_idt
            loss_idt_B = self.L1_norm_loss(idt_B, batch_A) * self.lambda_B * self.lambda_idt

            # Cycle loss
            loss_cyc_A = self.L1_norm_loss(rec_A, batch_A) * self.lambda_A
            loss_cyc_B = self.L1_norm_loss(rec_B, batch_B) * self.lambda_B

            # Combine losses & calculate gradients
            loss = loss_adv_A + loss_adv_B + loss_idt_A + loss_idt_B + loss_cyc_A + loss_cyc_B
            trainables = [x for x in self.G_A.trainable_variables]
            trainables += [x for x in self.G_B.trainable_variables]
            grads = tape.gradient(loss, trainables)
            self.optimizer_G.apply_gradients(zip(grads, trainables))

        return loss_adv_A, loss_adv_B, loss_idt_A, loss_idt_B, loss_cyc_A, loss_cyc_B


    def train_on_batch(self, batch_A, batch_B):
        # Train discriminators
        loss_D_A = self.train_discriminator(batch_A, batch_B, mode='B2A')
        loss_D_B = self.train_discriminator(batch_A, batch_B, mode='A2B')

        # Train Generators
        loss_adv_A, loss_adv_B, loss_idt_A, loss_idt_B, loss_cyc_A, loss_cyc_B = self.train_generator(batch_A, batch_B)

        loss_D = (loss_D_A + loss_D_B) / 2
        loss_G = loss_adv_A + loss_adv_B + loss_idt_A + loss_idt_B + loss_cyc_A + loss_cyc_B

        return loss_D, loss_G

    def evaluate(self, dataloader):
        loss_D_list = []
        loss_G_list = []
        for i, (batch_A, batch_B) in enumerate(dataloader.batch(1)):
            if i == 100: break

            # Generate samples
            fake_B = self.G_A(batch_A)  # A --> B
            fake_A = self.G_B(batch_B)  # B --> A
            idt_A = self.G_A(batch_B)  # B --> B
            idt_B = self.G_B(batch_A)  # B --> B
            rec_A = self.G_B(self.G_A(batch_A))  # A --> B --> A
            rec_B = self.G_A(self.G_B(batch_B))  # B --> A --> B

            # Discriminator loss
            loss_D_A = self.adversarial_loss(self.D_A(fake_A), False)
            loss_D_B = self.adversarial_loss(self.D_B(fake_B), False)

            # Adversarial loss
            loss_adv_A = self.adversarial_loss(self.D_B(fake_B), True)
            loss_adv_B = self.adversarial_loss(self.D_A(fake_A), True)

            # Identity loss
            loss_idt_A = self.L1_norm_loss(idt_A, batch_B) * self.lambda_A * self.lambda_idt
            loss_idt_B = self.L1_norm_loss(idt_B, batch_A) * self.lambda_B * self.lambda_idt

            # Cycle loss
            loss_cyc_A = self.L1_norm_loss(rec_A, batch_A) * self.lambda_A
            loss_cyc_B = self.L1_norm_loss(rec_B, batch_B) * self.lambda_B

            loss_D = (loss_D_A + loss_D_B) / 2
            loss_G = loss_adv_A + loss_adv_B + loss_idt_A + loss_idt_B + loss_cyc_A + loss_cyc_B

            loss_D_list.append(loss_D)
            loss_G_list.append(loss_G)

        return np.mean(loss_D_list), np.mean(loss_G_list)