import numpy as np
import torch
import math
import pandas as pd
import distortion
from tqdm import tqdm
from tools import *
import os

import optuna

def calc_psnr(X, Y):
    return 10 * np.log10(1 / ((X - Y)**2).mean())

metrics = {'psnr' : calc_psnr}

def run_params_grid(video_stream, filter_cls, params_grid, metric_name='psnr', verbose=True):
    metric = metrics[metric_name]

    video = np.stack(list(video_stream))
    df = pd.DataFrame.from_dict(dict_product(params_grid))
    metric_result = pd.Series()
    filter = filter_cls(video_stream.shape)

    print(df.shape)

    for idx, row in df.iterrows():
        params = row.to_dict()
        print('Parameters: ', params)
        try:
            filter.set_params(**params)
            filtered_video = np.stack([filter.apply_filter(frame) for i, frame in tqdm(enumerate(video), disable=not verbose)])
        except ValueError:
            print('blablablablablablablablablablablablablablablablablablablablablablablablablablabla')
            continue
        metric_result.loc[idx] = metric(video, filtered_video)

    df[metric_name] = metric_result
    
    return df


def distance(value, needed_value):
    return abs(value - needed_value)

def optuna_runner(video_stream, filter_cls, needed_values, calculated_df=pd.DataFrame(), metric_name='psnr', n_trials=100):
    filter = filter_cls(video_stream.shape)
    video = np.stack(list(video_stream))
    metric = metrics[metric_name]

    df = calculated_df.copy()
    df['type'] = 'calculated'

    def suggester(trial):
        res_params = {}
        for key, value in params_info.items():
            res_params[key] = suggest_type[key](trial, **params_info[key])
        return res_params

    def func(trial):
        nonlocal df
        params = suggester(trial)
        metric_value = df[(df[params.keys()] == params.values()).all(1)][metric_name]
        if metric_value.shape[0] != 0:
            return distance(metric_value.iloc[0], needed_value)
        try:
            filter.set_params(**params)
            filtered_video = np.stack([filter.apply_filter(frame) for frame in tqdm(video)])
            res = metric(video, filtered_video)
            params[metric_name] = res
            params['type'] = 'genetic'
            df = pd.concat([df, pd.DataFrame.from_dict(params, 'index').T])
            return distance(res, needed_value)
        except ValueError:
            print('blablablablablablablablablablablablablablablablablablablablablablablablablablabla')
            return float('inf')

    for needed_idx in range(len(needed_values)):
        needed_value = needed_values[needed_idx]
        first_gen = pd.concat([
            df[df[metric_name] > needed_value].sort_values(metric_name).iloc[:3], 
            df[df[metric_name] < needed_value].sort_values(metric_name).iloc[-3:]
            ])
        
        params = filter_cls.get_params_info()

        params_info = {}
        suggest_type = {}

        for key in params.keys():
            if params[key]['type'] == int:
                min_value = max(first_gen[key].min(), params[key]['range'][0])
                max_value = min(first_gen[key].max(), params[key]['range'][1])
                step_value = params[key]['range'][2]
                params_info[key] = {'name':key, 'low':min_value, 'high':max_value, 'step':step_value}
                suggest_type[key] = optuna.Trial.suggest_int
            elif params[key]['type'] == float:
                min_value = max(first_gen[key].min(), params[key]['range'][0])
                max_value = min(first_gen[key].max(), params[key]['range'][1])
                step_value = params[key]['range'][2]
                params_info[key] = {'name':key, 'low':min_value, 'high':max_value, 'step':step_value}
                suggest_type[key] = optuna.Trial.suggest_float
            else:
                params_info[key] = {'name':key, 'choices':first_gen[key].unique()}
                suggest_type[key] = optuna.Trial.suggest_categorical

        params_info_df = pd.DataFrame.from_dict(params_info)

        storage = optuna.storages.InMemoryStorage()

        study = optuna.study.create_study(
            sampler=optuna.samplers.QMCSampler(),
            direction='minimize',
            study_name="runner",
            storage=storage,
        )

        for idx, row in first_gen.iterrows():
            params = row[~(row.index == metric_name)].to_dict()
            study.enqueue_trial(params=params, user_attrs=row[metric_name])

        study.optimize(func, n_trials=len(first_gen))
        sampler = optuna.samplers.NSGAIISampler(population_size=50)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name="runner",
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(func, n_trials=n_trials, n_jobs=1)

        params = study.best_params
        params[metric_name] = df[(df[params.keys()] == params.values()).all(1)][metric_name].iloc[0]
        params['type'] = 'genetic'
        df = pd.concat([df, pd.DataFrame.from_dict(params, 'index').T])
        
    return df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--metric', type=str, default='psnr')
    parser.add_argument('--min_value', type=float, default=0)
    parser.add_argument('--max_value', type=float, default=100)
    first_args, _ = parser.parse_known_args()
    filter_cls = distortion.distortion_zoo[first_args.filter]
    for key, params in filter_cls.get_params_info().items():
        parser.add_argument(f'-{key}', f'--{key}',
                            type=params['type'], nargs='+', default=[params['default']])
    args, _ = parser.parse_known_args()

    metric_name = args.metric
    reader = y4m_reader(args.path)
    parameters = vars(args)
    parameters = {key: value for key, value in parameters.items(
    ) if key in filter_cls.get_params_info().keys()}

    print(parameters)

    df = run_params_grid(reader, filter_cls, parameters, metric_name)
    df.to_csv(args.save, index=False)

    df = df[~ (df[metric_name]==float('inf'))]
    df = df.sort_values(metric_name)
    hover_data = df.columns[~df.columns.isin([metric_name, 'type'])]

    needed_values, grid_values= create_grid(df[metric_name], n = 10, p = 0.60, min_value=args.min_value, max_value=args.max_value)
    res_df = optuna_runner(reader, filter_cls, needed_values, df, n_trials=25)
    res_df = res_df.drop_duplicates().reset_index()
    new_needed_values, new_grid_values = create_grid(res_df[metric_name], n = 10, p = 0.60, min_value=args.min_value, max_value=args.max_value)
    res_df.to_csv(args.save, index=False)
    present_video(filter_cls, reader, res_df[res_df[metric_name].isin(new_grid_values)], metric_name, f'{first_args.filter}_{os.path.splitext(os.path.basename(args.path))[0]}')


    
