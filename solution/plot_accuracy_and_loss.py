"""Parse json and plot accuracy and loss graphs."""
import os
import json
import argparse

import matplotlib.pyplot as plt

from common import FIGURES_DIR


# Arguments
def parse_args():
    """Parse script arguments."""

    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--model', '-m',
                        default='XceptionBased', type=str)
    parser.add_argument('--train_details_json', '-j',
                        default='out/AD_CTW_128_XceptionBased_Adam.json', type=str,
                        help='Json containing loss and accuracy.')
    parser.add_argument('--dataset', '-d',
                        default='AD_CTW_128', type=str)
    parser.add_argument('--patch_size',
                        default=128, type=int,
                        help='patch size: 128, 64, etc.')

    return parser.parse_args()


def main():
    """Parse script arguments, read json and plot accuracy and loss graphs."""
    args = parse_args()

    with open(args.train_details_json, mode='r', encoding='utf-8') as json_f:
        results_dict = json.load(json_f)[-1]

    losses_plot = plt.figure()
    plt.plot(range(1, len(results_dict['train_loss']) + 1),
             results_dict['train_loss'])
    plt.plot(range(1, len(results_dict['val_loss']) + 1),
             results_dict['val_loss'])
    plt.plot(range(1, len(results_dict['test_loss']) + 1),
             results_dict['test_loss'])
    plt.legend(['train', 'val', 'test'])
    plt.title(f'CE Loss vs Epoch for {args.model} Model on {args.patch_size} x {args.patch_size} Patches')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.grid(True)
    losses_plot.set_size_inches((8, 8))
    losses_plot.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_losses_plot.png'))

    accuracies_plot = plt.figure()
    plt.plot(range(1, len(results_dict['train_acc']) + 1),
             results_dict['train_acc'])
    plt.plot(range(1, len(results_dict['val_acc']) + 1),
             results_dict['val_acc'])
    plt.plot(range(1, len(results_dict['test_acc']) + 1),
             results_dict['test_acc'])
    plt.legend(['train', 'val', 'test'])
    plt.title(f'accuracy vs epoch for {args.model} '
              f'model on {args.dataset} dataset')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    accuracies_plot.set_size_inches((8, 8))
    accuracies_plot.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_accuracies_plot.png'))



if __name__ == '__main__':
    main()
