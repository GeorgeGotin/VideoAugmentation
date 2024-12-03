import pyiqa
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pyiqa import imread2tensor
import numpy as np
import shutil
import subprocess
import os
import sys
import seaborn as sns
from functools import partial
from scipy.stats import spearmanr
import torch.nn as nn
import cv2
import numpy as np
import albumentations
import optuna
from itertools import product

from tools_io import y4m_reader, y4m_writer

# много лишнего

# Function to read YUV 4:2:0 frame and convert it to RGB


def dict_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


def parse_psnr_avg(file_path):
    psnr_values = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                val = float(line.split('psnr_avg:')[1].split()[0])
            except Exception:
                val = 1000
            psnr_values.append(val)

    if psnr_values:
        mean_psnr = sum(psnr_values) / len(psnr_values)
        return mean_psnr
    else:
        return None


def start_optimization(
    objective_func,  # принимает trial, X_tr, y_tr, X_val, y_val, **other_objective_kwargs
    n_trials,
    n_jobs,
    study_direction=None,
    sampler=None,
    features=None,
    **other_objective_kwargs
):
    '''
    function, performing optina optimization with given objective function. 
    also it filters data and scales it.
    '''

    obj_func = objective_func
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(obj_func, n_trials=n_trials, n_jobs=n_jobs)
    return study


def sample_yuv_frames(input_path, output_file, width, height, num_frames=15):
    y_size = width * height
    u_size = y_size // 4
    frame_size = y_size + 2 * u_size
    with open(output_file, 'wb') as out_file:
        with open(input_path, 'rb') as file:
            file.seek(0, 2)
            file_size = file.tell()
            total_frames = file_size // frame_size
            num_frames = min(num_frames, total_frames)
            interval = total_frames // num_frames
            file.seek(0)
            for i in range(num_frames):
                frame_index = i * interval
                file.seek(frame_index * frame_size)
                y = file.read(y_size)
                u = file.read(u_size)
                v = file.read(u_size)

                out_file.write(y)
                out_file.write(u)
                out_file.write(v)

    print(f"Saved {num_frames} frames to {output_file} in YUV format")


def psnr_analyze(psnr_file, max_substract=0, max_add=3, distance_threshold=None):
    psnr_values = []

    with open(psnr_file, "r") as file:
        for line in file:
            start = line.find("psnr=")
            if start != -1:
                psnr_str = line[start + 5:]
                try:
                    psnr_values.append(float(psnr_str))
                except ValueError:
                    pass
    psnr_values = sorted(psnr_values)
    if distance_threshold is None:
        distance_threshold = 4 * (psnr_values[-1] - psnr_values[0])/len(psnr_values)
    # maybe do something for substracting later
    needed_psnrs = psnr_values
    added_psnrs = []
    while len(added_psnrs) < max_add - max_substract:
        change_idx = max(range(len(needed_psnrs) - 1),
                         key=lambda x: needed_psnrs[x + 1] - needed_psnrs[x])
        val = needed_psnrs[change_idx + 1] - needed_psnrs[change_idx]
        if val < distance_threshold:
            break
        while needed_psnrs[change_idx] not in psnr_values:
            change_idx -= 1
        amount = 1
        start_idx = change_idx
        change_idx += 1
        while needed_psnrs[change_idx] not in psnr_values:
            amount += 1
            change_idx += 1
        end_idx = change_idx
        new_psnrs = []
        for i in range(start_idx + 1):
            new_psnrs.append(needed_psnrs[i])
        for i in range(amount):
            val = needed_psnrs[start_idx] + \
                (i + 1)*(needed_psnrs[end_idx] - needed_psnrs[start_idx])/(amount + 1)
            new_psnrs.append(val)
        for i in range(end_idx, len(needed_psnrs)):
            new_psnrs.append(needed_psnrs[i])
        needed_psnrs = new_psnrs
    for x in needed_psnrs:
        if x not in psnr_values:
            added_psnrs.append(x)

    plt.scatter(psnr_values, [0] * len(psnr_values), color="blue",
                s=100)  # 's' is the size of the dots
    plt.scatter(added_psnrs, [0] * len(added_psnrs), color="red", s=100)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.5)  # Draw a horizontal line

    plt.ylim(-1, 1)
    plt.xlim(min(psnr_values) - 1, max(psnr_values) + 1)
    plt.show()
    return added_psnrs
