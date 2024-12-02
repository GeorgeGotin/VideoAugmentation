import numpy as np
import torch
import pandas as pd
import distortion
from tqdm import tqdm
from tools import *


def run_params_grid(video_reader, filter, params_grid, verbose=True):

    df = pd.DataFrame.from_dict(dict_product(params_grid))
    psnr_array = [[] for _ in range(len(df))]

    def calc(row):
        params = row.to_dict()
        filter.set_params(**params)
        frame2 = filter.apply_filter(frame)
        delta = ((frame - frame2) ** 2).mean()
        psnr_array[row.name].append(delta)

    for i, (frame, _) in tqdm(enumerate(video_reader), disable=not verbose):
        df.apply(calc, axis=1)

    psnr_array = np.array(psnr_array)

    df['psnr'] = 10 * np.log10(255**2 / psnr_array.mean(axis=1))

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str)
    parser.add_argument('--path', type=str)
    first_args, _ = parser.parse_known_args()
    fltr = distortion.distortion_zoo[first_args.filter]()
    for key, (dtype, _, _) in fltr.get_params_info().items():
        parser.add_argument(f'-{key}', f'--{key}', type=dtype, nargs='+')
    args, _ = parser.parse_known_args()

    reader = y4m_reader(args.path)
    parameters = vars(parser)

    df = run_params_grid()
