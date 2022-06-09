""" A script for pixel wise segmentation using sliding window
    with network predictions"""
import os
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import torch
from torchvision import transforms
from utils import load_model
from math import exp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def segmentation_nn_window(input_patch_path, res, patch_size):
    """ Function which returns the segmentation image as an array"""

    input_patch = Image.open(input_patch_path)
    input_size = patch_size - res + 1

    # Defining np 2d arrays which will contain final predictions, pixel scores and number of prediction on each pixel
    segmentation_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
    sum_defected_scores = np.zeros((patch_size, patch_size))
    nof_predictions = np.zeros((patch_size, patch_size))

    # Loading data in small windows/patches and running xception_based model on each one
    for y in range(patch_size - res + 1):
        input_data = torch.zeros((input_size, 1, res, res), dtype=torch.float32)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.4,
                                 0.2437)])
        input_idx = 0
        for x in range(patch_size - res + 1):
            bounds = (x, y, x + res, y + res)
            im_crop = input_patch.crop(bounds)
            im_crop_t = transform(im_crop)
            input_data[input_idx] = im_crop_t
            input_idx += 1
        if y == 0:
            model = load_model()
            model.load_state_dict(torch.load(os.path.join(os.getcwd(), "checkpoints/AD_CTW_X5_"
                                                                       "%d_XceptionBased_Adam.pt" % res)))
            model.eval()
        with torch.no_grad():
            input_data = input_data.to(device)
            pred = model(input_data)
            pred = pred.to(device)

            # Calculating scores
            for x in range(patch_size - res + 1):
                d_score = exp(pred[x, 1]) / (exp(pred[x, 0]) + exp(pred[x, 1]))
                sum_defected_scores[y:y + res, x:x + res] += d_score
                nof_predictions[y:y + res, x:x + res] += 1

    # Calculating final average scores
    avg_defected_scores = sum_defected_scores / nof_predictions

    # Defining a 2d np array containing data on each pixel nearest neighbors
    likely_defected_neighbours = np.zeros((patch_size, patch_size))

    for y in range(patch_size):
        for x in range(patch_size):

            #Checking how many "suspect" neighbors each pixel has
            if (y > res) and (y < patch_size - res) and (x > res) and (x < patch_size - res):
                if avg_defected_scores[y, x] > 0.925:
                    if avg_defected_scores[y + 1, x] > 0.925:
                        likely_defected_neighbours[y, x] += 1
                    if avg_defected_scores[y, x + 1] > 0.925:
                        likely_defected_neighbours[y, x] += 1
                    if avg_defected_scores[y - 1, x] > 0.925:
                        likely_defected_neighbours[y, x] += 1
                    if avg_defected_scores[y, x - 1] > 0.925:
                        likely_defected_neighbours[y, x] += 1

                    # Making final prediction
                    if likely_defected_neighbours[y, x] <= 1 and avg_defected_scores[y, x] > 0.94:
                        segmentation_patch[y, x] = 255
                    elif likely_defected_neighbours[y, x] <= 3 and avg_defected_scores[y, x] > 0.93:
                        segmentation_patch[y, x] = 255
                    elif likely_defected_neighbours[y, x] == 4:
                        segmentation_patch[y, x] = 255

    return segmentation_patch


def eval_results(segmentation_patch, blobs_array, patch_size, res):
    """Function which evaluates the results and returns summary of predictions"""

    eval_arr = np.zeros([patch_size, patch_size], dtype=float)

    total_pixels = 0
    correct_defect_labeled_pixels = 0
    correct_clean_labeled_pixels = 0
    total_defect_labeled_pixels = 0
    total_defect_pixels = 0

    for y in range(patch_size):
        for x in range(patch_size):
            if not (x <= res or y <= res or x >= patch_size - res or y >= patch_size - res):
                total_pixels += 1
                if blobs_array[y, x] != 0:
                    total_defect_pixels += 1
                if segmentation_patch[y, x] != 0:
                    total_defect_labeled_pixels += 1
                    if segmentation_patch[y, x] == blobs_array[y, x]:   # True Positive prediction
                        eval_arr[y, x] = 1
                        correct_defect_labeled_pixels += 1
                    else:   # False positive prediction
                        eval_arr[y, x] = 0.67
                if segmentation_patch[y, x] == 0:
                    if segmentation_patch[y, x] == blobs_array[y, x]:   # True Negative prediction
                        correct_clean_labeled_pixels += 1
                    else:   # False Negative prediction
                        eval_arr[y, x] = 0.4
    correct_labeled_pixels = correct_defect_labeled_pixels + correct_clean_labeled_pixels
    return eval_arr, total_pixels, correct_labeled_pixels, correct_defect_labeled_pixels, total_defect_labeled_pixels, total_defect_pixels


if __name__ == '__main__':
    patch_size = 256
    res = 16

    all_total_pixels = 0
    all_total_defected_pixels = 0
    all_correct_labeled_pixels = 0
    all_defect_labeled_pixels = 0
    all_correct_defect_labeled_pixels = 0

    blobs_patch_path = "Insert blobs map path"
    input_patch_path = "Insert input image with size (256 x 256) path"

    blobs_patch = Image.open(blobs_patch_path)
    blobs_array = np.array(blobs_patch)

    # Plotting real blobs map
    cmap = cm.get_cmap('hot')
    plt.plot(0, 0, "-", color=cmap(0.9), label="Defect")
    plt.plot(0, 0, "-", color=cmap(0), label="Clean")
    plt.imshow(blobs_array, cmap=cmap)
    plt.title('Actual Defects on wafer', fontsize=18)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 0.87), fontsize=12)
    plt.show()

    # Computing segmentation image
    segmentation_patch = segmentation_nn_window(input_patch_path, res, patch_size)

    # Evaluating results
    eval_arr, total_pixels, correct_labeled_pixels, correct_defect_labeled_pixels, total_defect_labeled_pixels, total_defect_pixels = \
        eval_results(segmentation_patch, blobs_array, patch_size, res)

    # Plotting segmentation map
    cmap = cm.get_cmap('hot')
    plt.plot(0, 0, "-", color=cmap(0.9), label="Defect")
    plt.plot(0, 0, "-", color=cmap(0), label="Clean")
    plt.imshow(segmentation_patch, cmap=cmap)
    plt.title('Segmentation Image', fontsize=18)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 0.87), fontsize=12)
    plt.show()

    # Plotting predictions summary
    plt.plot(0, 0, "-", color=cmap(0.9), label="TP")
    plt.plot(0, 0, "-", color=cmap(0), label="TN")
    plt.plot(0, 0, "-", color=cmap(0.67), label="FP")
    plt.plot(0, 0, "-", color=cmap(0.4), label="FN")
    plt.imshow(eval_arr, cmap=cmap)
    plt.title('Predictions summary (zoom in on defects to see better)', fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 0.87), fontsize=12)
    plt.show()

    # Printing a summary of the results
    accuracy = correct_labeled_pixels / total_pixels
    precision = 0
    detection_rate = 0
    if total_defect_labeled_pixels != 0:
        precision = correct_defect_labeled_pixels / total_defect_labeled_pixels
    if total_defect_pixels != 0:
        detection_rate = correct_defect_labeled_pixels / total_defect_pixels
        print(
            f'Acc: {accuracy:.5f}[%]  '
            f'({correct_labeled_pixels}/{total_pixels}) '
            f'Precision: {precision:.5f}[%]  '
            f'({correct_defect_labeled_pixels}/{total_defect_labeled_pixels}) '
            f'Detection Rate: {detection_rate:.5f}[%]  '
            f'({correct_defect_labeled_pixels}/{total_defect_pixels}) '
        )
