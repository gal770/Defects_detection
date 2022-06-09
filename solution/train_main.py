import argparse

import torch
from torch import optim

from utils import load_dataset, load_model, load_loss
from trainer import LoggingParameters, Trainer

"""Main training script."""


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Arguments
def parse_args():
    """Parse script arguments.

    Get training hyper-parameters such as: learning rate, momentum,
    batch size, number of training epochs and optimizer.
    Get training dataset and the model name.
    """
    parser = argparse.ArgumentParser(description='Training models with Pytorch')
    parser.add_argument('--lr_start', default=0.001, type=float,
                        help='learning rate starting value')
    parser.add_argument('--lr_decay', default=0.95, type=float,
                        help='learning rate decay rate')
    parser.add_argument('--optimizer', '-o', default='Adam',
                        type=str, help='Optimization Algorithm')
    parser.add_argument('--batch_size', '-b', default=64, type=int,
                        help='Training batch size')
    parser.add_argument('--epochs', '-e', default=1, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--dataset', '-d',
                        default='AD_CTW_X5_16', type=str)
    parser.add_argument('--patch_size',
                        default=16, type=int,
                        help='patch size: 128, 64, etc.')

    return parser.parse_args()


def main():
    """Parse arguments and train model on dataset."""
    args = parse_args()
    # Data
    print(f'==> Preparing data: {args.dataset.replace("_", " ")}..')

    train_dataset = load_dataset(dataset_name=args.dataset,
                                 dataset_part='train', patch_size=args.patch_size)
    val_dataset = load_dataset(dataset_name=args.dataset, dataset_part='val', patch_size=args.patch_size)
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test', patch_size=args.patch_size)

    # Model
    model_name = 'XceptionBased'
    model = load_model()

    # Loss
    loss_name = 'CE_loss'
    criterion = load_loss()

    # Build optimizer
    optimizers = {
        'Adam': lambda: optim.Adam(model.parameters(), lr=args.lr_start),
    }

    optimizer_name = args.optimizer
    if optimizer_name not in optimizers:
        raise ValueError(f'Invalid Optimizer name: {optimizer_name}')

    #print(f"Building optimizer {optimizer_name}...")
    optimizer = optimizers[args.optimizer]()
    #print(optimizer)

    optimizer_params = optimizer.param_groups[0].copy()
    # remove the parameter values from the optimizer parameters for a cleaner
    # log
    del optimizer_params['params']

    # schedule exponential learning rate
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=- 1)

    # Batch size
    batch_size = args.batch_size

    # Training Logging Parameters
    logging_parameters = LoggingParameters(model_name=model_name,
                                           dataset_name=args.dataset,
                                           optimizer_name='Adam',
                                           optimizer_params=optimizer_params)

    # Create an abstract trainer to train the model with the data and parameters
    # above:
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      criterion=criterion,
                      batch_size=batch_size,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      test_dataset=test_dataset)

    # Train, evaluate and test the model:
    trainer.run(epochs=args.epochs, logging_parameters=logging_parameters)


if __name__ == '__main__':
    main()