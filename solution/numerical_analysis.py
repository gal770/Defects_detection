"""Plot Precision-Detection Rate curve with average precision score"""
import os
import argparse

import torch
import scipy.stats as sp
import matplotlib.pyplot as plt

from sklearn import metrics
from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.
    Returns:
    Dataset and patch size"""

    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--dataset', '-d',
                        default='AD_CTW_X5_16', type=str)
    parser.add_argument('--patch_size',
                        default=16, type=int,
                        help='patch size: 128, 64, etc.')

    return parser.parse_args()


def get_soft_scores_and_true_labels(dataset, model):
    """Return the soft scores and ground truth labels for the dataset.

    Loop through the dataset (in batches), log the model's soft scores for
    all samples in two iterables: all_first_soft_scores and
    all_second_soft_scores. Log the corresponding ground truth labels in
    gt_labels.

    Args:
        dataset: the test dataset to scan.
        model: the model used to compute the prediction.

    Returns:
        (all_first_soft_scores, all_second_soft_scores, gt_labels):
        all_first_soft_scores: an iterable holding the model's first
        inference result on the images in the dataset (data in index = 0).
        all_second_soft_scores: an iterable holding the model's second
        inference result on the images in the dataset (data in index = 1).
        gt_labels: an iterable holding the samples' ground truth labels.
    """

    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=True)
    all_first_soft_scores = torch.zeros(0)
    all_first_soft_scores = all_first_soft_scores.to(device)
    all_second_soft_scores = torch.zeros(0)
    all_second_soft_scores = all_second_soft_scores.to(device)
    gt_labels = torch.zeros(0, dtype=int)
    gt_labels = gt_labels.to(device)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        with torch.no_grad():
            pred = model(inputs)
            pred = pred.to(device)
            targets = targets.to(device)
            all_first_soft_scores_batch = pred[:, 0]
            all_second_soft_scores_batch = pred[:, 1]
            gt_labels_batch = targets

            all_first_soft_scores = torch.cat((all_first_soft_scores, all_first_soft_scores_batch))
            all_second_soft_scores = torch.cat((all_second_soft_scores, all_second_soft_scores_batch))
            gt_labels = torch.cat((gt_labels, gt_labels_batch))

    return all_first_soft_scores, all_second_soft_scores, gt_labels

def plot_recall_precision_curve(recall_precision_curve_figure,
                   all_first_soft_scores,
                   all_second_soft_scores,
                   gt_labels, args):
    """Plot a recall-precision curve for the two scores on the given figure.

    Args:
        recall_precision_curve_figure: the figure to plot on.
        all_first_soft_scores: iterable of soft scores.
        all_second_soft_scores: iterable of soft scores.
        gt_labels: ground truth labels.

    Returns:
        recall_precision_curve_first_score_figure: the figure with plots on it.
    """
    precision, recall, _ = metrics.precision_recall_curve(gt_labels.cpu(), all_second_soft_scores.cpu())
    plt.plot(recall, precision)
    plt.grid(True)
    plt.xlabel('Detection rate (defects detected / all defects)',fontsize=16)
    plt.ylabel('Precision (actual defects / predicted defects)', fontsize=16)
    plt.suptitle(f'Classification: Precision vs Detection Rate for {args.patch_size} x {args.patch_size} patches', fontsize=16)
    plt.title(f'Average Precision score: {metrics.average_precision_score(gt_labels, all_second_soft_scores):.5f}', fontsize=16)
    recall_precision_curve_figure.set_size_inches((8, 8))
    return recall_precision_curve_figure



def main():
    """Parse script arguments, log all the model's soft scores on the dataset
    images and the true labels. Use the soft scores and true labels to
    generate Precision-Detection Rate curve with average precison score"""
    args = parse_args()

    # load model
    model = load_model()
    model.load_state_dict(
        torch.load("/home/eeproj5/Documents/solution_camtek/solution/checkpoints/AD_CTW_%d_XceptionBased_Adam.pt" % args.patch_size))
    model.eval()

    # load dataset
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test', patch_size=128)
    all_first_soft_scores, all_second_soft_scores, gt_labels = \
        get_soft_scores_and_true_labels(test_dataset, model)

    # plot the recall-precision curves
    recall_precision_curve_figure = plt.figure()
    roc_curve_figure = plot_recall_precision_curve(recall_precision_curve_figure,
                                      all_first_soft_scores.cpu(),
                                      all_second_soft_scores.cpu(),
                                      gt_labels.cpu(), args)
    roc_curve_figure.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_recall_precision_curve.png'))




if __name__ == '__main__':
    main()