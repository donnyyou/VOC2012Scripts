#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(yas@meitu.com)


import numpy as np

def get_iou_accuracy(res1, res2):
    if np.sum(np.logical_or(res1, res2)) == 0:
        return -0.1
    return float(np.sum(np.logical_and(res1, res2))) / np.sum(np.logical_or(res1, res2))

def get_iou_list(res1, res2, num_classes):
    iou_list = list()
    for i in range(num_classes):
        # relabel
        result1 = np.copy(res1)
        result2 = np.copy(res2)
        if i == 0:
            result1[result1 != 0] = 2
            result1[result1 == 0] = 1
            result1[result1 == 2] = 0
            
            result2[result2 != 0] = 2
            result2[result2 == 0] = 1
            result2[result2 == 2] = 0
        else:
            result1[result1 != i] = 0 
            result1[result1 == i] = 1
            result2[result2 != i] = 0
            result2[result2 == i] = 1
        iou = get_iou_accuracy(result1, result2)
        iou_list.append(iou)

    return iou_list


if __name__ == "__main__":
    pass
