import torch
import copy
import numpy as np
import math

def square_poison(data, label, target_label, attack_ratio=0.1):
    data = copy.deepcopy(data)
    label = copy.deepcopy(label)

    target_tensor = []
    poison_number = math.floor(len(label) * attack_ratio)

    trigger_value = 255/255  #10/255
    pattern_type = [[[1, 1], [1, 2]],
                    [[2, 1], [2, 2]]]

    for index in range(poison_number):
        label[index] = target_label
        for channel in range(1):
            for i in range(len(pattern_type)):
                for j in range(len(pattern_type[i])):
                    pos = pattern_type[i][j]
                    data[index][channel][pos[0]][pos[1]] = trigger_value

    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    label = label[random_perm]

    return data, label
