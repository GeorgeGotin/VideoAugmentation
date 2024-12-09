import numpy as np
import torch
import pandas as pd
import distortion
from tqdm import tqdm
from tools import *

import optuna


def run_params_grid(video_stream, filter_cls, params_grid, verbose=True):

    df = pd.DataFrame.from_dict(dict_product(params_grid))
    psnr_array = [[] for _ in range(len(df))]
    filter = filter_cls(video_stream.shape)

    def calc(row):
        params = row.to_dict()
        filter.set_params(**params)
        frame2 = filter.apply_filter(frame)
        delta = ((frame - frame2) ** 2).mean()
        psnr_array[row.name].append(delta)

    for i, frame in tqdm(enumerate(video_stream), disable=not verbose):
        df.apply(calc, axis=1)

    psnr_array = np.array(psnr_array)

    df['psnr'] = 10 * np.log10(1**2 / psnr_array.mean(axis=1))

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str)
    parser.add_argument('--path', type=str)
    first_args, _ = parser.parse_known_args()
    filter_cls = distortion.distortion_zoo[first_args.filter]
    for key, params in filter_cls.get_params_info().items():
        parser.add_argument(f'-{key}', f'--{key}',
                            type=params['type'], nargs='+', default=params['default'])
    args, _ = parser.parse_known_args()

    reader = y4m_reader(args.path)
    parameters = vars(args)
    parameters = {key: value for key, value in parameters.items(
    ) if key in filter_cls.get_params_info().keys()}

    print(parameters)

    df = run_params_grid(reader, filter_cls, parameters)

    print(df.to_csv())
