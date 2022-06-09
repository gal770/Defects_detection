# Created By Biran Sider And Gal Setty
import sys
import os
import numpy as np
from scipy import ndimage as ndi
import imageio
import cv2
from matplotlib import pyplot as plt
import math


# Performing upscaling then downscaling.
def PyrOpp(ima, level):
    downsized = ima
    for i in range(1, level):
        downsized = cv2.pyrUp(downsized)
    sampleBased = downsized
    for i in range(1, level):
        sampleBased = cv2.pyrDown(sampleBased)
    return sampleBased


# Pyramid downscale then upscale.
def PyrLevel(ima, level):
    downsized = ima
    for i in range(1, level):
        downsized = cv2.pyrDown(downsized)
    sampleBased = downsized
    for i in range(1, level):
        sampleBased = cv2.pyrUp(sampleBased)
    return sampleBased


# Apply filter Sobel X.
def SobelX(image):
    sobel_gradX = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], np.float32)
    return ndi.convolve(image, sobel_gradX)


# The gradient based image partitioning.
def ImagePartitioning(image, step, n, grad=None):
    if (grad is None):
        grad = SobelX(image)
    (w, h) = image.shape
    x = w
    if (w < h):
        x = w
    if (x <= step):
        sub = np.bitwise_xor(image, PyrLevel(image, n)[-w:, -h:])
        return (sub).astype(np.uint8)
    else:
        part_point = np.argmax(np.sum(grad[step // 2:w - step // 2, :], axis=1)) + step // 2
        grad = SobelX(np.transpose(image))
        return np.concatenate((np.transpose(
            ImagePartitioning(np.transpose(image[:part_point, :]), step, n, grad[:, :part_point])), np.transpose(
            ImagePartitioning(np.transpose(image[part_point:, :]), step, n, grad[:, part_point:]))), axis=0)


# The processing over augmanted image (rotating it)
def Process_Image(image, step, pyr_level):
    mapped_division_1 = ImagePartitioning(image, step, pyr_level).reshape(image.shape)
    mapped_division_2 = np.transpose(ImagePartitioning(np.transpose(image), step, pyr_level)).reshape(image.shape)
    mapped_division_3 = np.transpose(ImagePartitioning(np.transpose(image[::-1, ::-1]), step, pyr_level))[::-1,
                        ::-1].reshape(image.shape)
    mapped_division_4 = ImagePartitioning(image[::-1, ::-1], step, pyr_level)[::-1, ::-1].reshape(image.shape)
    return np.bitwise_and(np.bitwise_and(mapped_division_1, mapped_division_2),
                          np.bitwise_and(mapped_division_3, mapped_division_4)).reshape(image.shape)


# The final implementation taking all partitioning methods and wegithing them
def Sizing_Sampling(image):
    Five_Levels = Process_Image(image, 64, 4)
    Four_Levels = Process_Image(image, 32, 4)
    cross_Five_Four = np.bitwise_and(Five_Levels, Four_Levels)

    Three_Levels = Process_Image(image, 16, 4)
    cross_Five_Three = np.bitwise_and(Five_Levels, Three_Levels)
    cross_Four_Three = np.bitwise_and(Four_Levels, Three_Levels)
    Five_To_Three = np.bitwise_or(cross_Five_Four, np.bitwise_or(cross_Five_Three, cross_Four_Three))

    Two_Levels = Process_Image(image, 8, 3)
    cross_Five_Two = np.bitwise_and(Five_Levels, Two_Levels)
    cross_Four_Two = np.bitwise_and(Four_Levels, Two_Levels)
    cross_Three_Two = np.bitwise_and(Three_Levels, Two_Levels)
    Five_To_Two = np.bitwise_or(Five_To_Three,
                                np.bitwise_or(cross_Five_Two, np.bitwise_or(cross_Four_Two, cross_Three_Two)))

    One_Level = Process_Image(image, 4, 3)
    cross_Five_One = np.bitwise_and(Five_Levels, One_Level)
    cross_Four_One = np.bitwise_and(Four_Levels, One_Level)
    cross_Three_One = np.bitwise_and(Three_Levels, One_Level)
    cross_Two_One = np.bitwise_and(Two_Levels, One_Level)
    fin_One = np.bitwise_or(cross_Five_One,
                            np.bitwise_or(cross_Four_One, np.bitwise_or(cross_Three_One, cross_Two_One)))
    return np.bitwise_or(fin_One, Five_To_Two)


# The primitive size based partitioning algorithm
def PartitioningGeneral(image, step, n):
    (w, h) = image.shape
    if (w == h):
        if (n > math.log2(step)):
            n = np.uint8(math.log2(step))
        if h == step:
            return np.bitwise_xor(image, PyrLevel(image, n))
        else:
            return np.uint8(np.bitwise_xor(image, PyrLevel(image, n)) * 0.4 + 0.6 * np.concatenate((np.transpose(
                PartitioningGeneral(np.transpose(image[:np.uint8(w / 2), :]), step, n)), np.transpose(
                PartitioningGeneral(np.transpose(image[np.uint8(w / 2):, :]), step, n))), axis=0).reshape(image.shape))
    else:
        return np.concatenate((np.transpose(PartitioningGeneral(np.transpose(image[np.uint8(w / 2):, :]), step, n)),
                               np.transpose(PartitioningGeneral(np.transpose(image[:np.uint8(w / 2), :]), step, n))),
                              axis=0).reshape(image.shape)


# The upscaled based soloution:
def CompareExpendedReduced(image):
    Four = np.bitwise_and(np.bitwise_xor(PyrLevel(image, 4), image), np.bitwise_xor(PyrOpp(image, 4), image))
    Three = np.bitwise_and(np.bitwise_xor(PyrLevel(image, 3), image), np.bitwise_xor(PyrOpp(image, 3), image))
    Two = np.bitwise_and(np.bitwise_xor(PyrLevel(image, 2), image), np.bitwise_xor(PyrOpp(image, 2), image))
    return np.bitwise_or(np.bitwise_and(Two, Three),
                         np.bitwise_or(np.bitwise_and(Four, Three), np.bitwise_and(Four, Two)))