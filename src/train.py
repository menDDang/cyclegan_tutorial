from model import cycleGAN
from utils import hparams
from utils import dataloader
import tensorflow as tf
import numpy as np
import argparse
import os


if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_v2.yaml', help='configuration file')
    parser.add_argument('--datadir', type=str, default='datasets/vangogh2photo', help='directory path of data')
    args = parser.parse_args()

    # Set hyper parameters
    config = args.config
    hp = hparams.HParam(config)

    # Create summary writer
    writer = tf.summary.create_file_writer(logdir=hp.train.log_dir)

    # Create data loaders
    train_data_loader = dataloader.create_data_loader(hp, train=True)
    test_data_loader = dataloader.create_data_loader(hp, train=False)

    # Build model
    model = cycleGAN.CycleGAN(hp)

    # Do training
    for epoch, (batch_A, batch_B) in enumerate(train_data_loader.batch(hp.train.batch_size)):
        loss_D, loss_G = model.train_on_batch(batch_A, batch_B)

        if epoch % hp.train.summary_interval == 0:
            with writer.as_default():
                tf.summary.scalar('Train Loss D', loss_D, step=epoch)
                tf.summary.scalar('Train Loss G', loss_G, step=epoch)

        if epoch % hp.train.save_chkpt_interval == 0:
            chkpt_name = os.path.join(hp.train.chkpt_dir, 'chkpt-' + str(epoch))
            model.save_weights(chkpt_name)

        if epoch % hp.train.loss_evaluation_interval == 0:
            test_loss_D, test_loss_G = model.evaluate(test_data_loader)
            with writer.as_default():
                tf.summary.scalar('Test Loss D', test_loss_D, step=epoch)
                tf.summary.scalar('Test Loss G', test_loss_G, step=epoch)

        if epoch % hp.train.sample_evaluation_interval == 0:
                # evaluate using generated samples
            for A, B in test_data_loader.batch(1):
                fakeB = model.G_A(A)  # A --> B
                fakeA = model.G_B(B)  # B --> A

                recA = model.G_B(fakeB) # A --> B --> A
                recB = model.G_A(fakeA) # B --> A --> B

                size = hp.data.size
                sample_A = np.zeros(shape=[1, size, 3*size, 3])
                sample_A[0, :, :size,:] = np.array(A)
                sample_A[0, :, size:size*2, :] = np.array(fakeB)
                sample_A[0, :, size*2:, :] = np.array(recA)

                sample_B = np.zeros(shape=[1, size, 3*size, 3])
                sample_B[0, :, :size,:] = np.array(B)
                sample_B[0, :, size:size*2, :] = np.array(fakeA)
                sample_B[0, :, size*2:, :] = np.array(recB)

                with writer.as_default():
                    tf.summary.image('A', sample_A, step=epoch)
                    tf.summary.image('B', sample_B, step=epoch)
                break  # do not erase!!!!

        epoch += 1

        print("Epoch : {}, Train Loss D : {}, Train Loss G : {}".format(epoch, "%1.3f" % loss_D, "%1.3f" % loss_G))

        if epoch >= hp.train.train_epoch_num: break

    print("Optimization is Done!")