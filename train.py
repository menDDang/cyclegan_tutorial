from model import cycleGAN
from utils import hparams
import argparse

if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='configuration file')
    parser.add_argument('--datadir', type=str, default='datasets/vangogh2photo', help='directory path of data')
    args = parser.parse_args()

    # Set hyper parameters
    config = args.config
    hp = hparams.HParam(config)

    # Create data loaders
    train_data_loader =
    test_data_loader =

    # Build model
    model = cycleGAN.CycleGAN(hp)

