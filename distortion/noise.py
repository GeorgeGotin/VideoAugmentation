from .abstract import Abstract
import albumentations as A
import numpy as np


class AbstractNoise(Abstract):
    def __init__(self, shape):
        super().__init__(shape)

    def set_params(self, temporal_mode, **params):
        params['temporal_mode'] = temporal_mode
        super().set_params(**params)

    def apply_filter(self, frame):

        frame = frame.copy()
        if self.temporal_mode == 'constant':
            return np.clip(frame + self.mask, 0, 1)

        return np.clip(self.filter(image=frame)['image'], 0, 1)

    def get_objective(self, trial, **rest):
        return super().get_objective(trial, **rest)


class UniformNoise(AbstractNoise):

    def make_mask(self):
        self.mask = np.random.rand(*self.shape)
        self.mask *= [self.delta_r, self.delta_g, self.delta_b]

    def set_params(self, delta_r, delta_g, delta_b, overlay='additive', temporal_mode='constant', *args, **kwargs):

        super().set_params(temporal_mode, **
                           dict(zip(['delta_r', 'delta_g', 'delta_b', 'overlay'], [delta_r, delta_g, delta_b, overlay])))

        # self.filter = A.transforms.AdditiveNoise(
            # noise_type='uniform', p=1.0, spatial_mode='per_pixel', noise_params={'ranges': [(-delta_r, delta_r), (-delta_g, delta_g), (-delta_b, delta_b)]})

        if self.temporal_mode == 'constant':
            self.make_mask()

    def apply_filter(self, frame):
        frame = frame.copy()
        if self.temporal_mode == 'per_frame':
            self.make_mask()
        if self.overlay == 'additive':
            return np.clip(frame + self.mask, 0, 1)
        if self.overlay == 'multiplicative':
            return np.clip(frame * self.mask, 0, 1)

    def get_params(self):
        return {s: self.__getattribute__(s) for s in ['temporal_mode', 'delta_r', 'delta_g', 'delta_b']}

    @staticmethod
    def get_params_info():
        info = {
            'overlay' : dict(type=str, range=['additive', 'multiplicative'], default='additive', info='Mode of overlaying noise over frame'),
            'temporal_mode': {'type': str, 'range': ['constant', 'per_frame'], 'info': 'Mode of applying noise. contant - one for all frames, per_frame - individual noise for each frame', 'default': 'per_frame'},
            'delta_r': {'type': float, 'range': [0, 1, None], 'info': 'Maximum deviation for red channel', 'default': 0.0},
            'delta_g': {'type': float, 'range': [0, 1, None], 'info': 'Maximum deviation for green channel', 'default': 0.0},
            'delta_b': {'type': float, 'range': [0, 1, None], 'info': 'Maximum deviation for blue channel', 'default': 0.0},
        }
        return info

class UniformConnectedNoise(UniformNoise):
    def set_params(self, delta, overlay='additive', temporal_mode='constant', *args, **kwargs):
        super().set_params(delta_b=delta, delta_g=delta, delta_r=delta,temporal_mode=temporal_mode,overlay=overlay, *args, **kwargs)

    @staticmethod
    def get_params_info():
        info = {
            'overlay' : dict(type=str, range=['additive', 'multiplicative'], default='additive', info='Mode of overlaying noise over frame'),
            'temporal_mode': {'type': str, 'range': ['constant', 'per_frame'], 'info': 'Mode of applying noise. contant - one for all frames, per_frame - individual noise for each frame', 'default': 'per_frame'},
            'delta': {'type': float, 'range': [0, 1, None], 'info': 'Maximum deviation for each channel', 'default': 0.0},
        }
        return info

class GaussianNoise(AbstractNoise):
    def set_params(self, mean_min, mean_max, std_min, std_max, temporal_mode, *args, **kwargs):
        super().set_params(temporal_mode, **
                           dict(zip(['mean_min', 'mean_max', 'std_min', 'std_max'], [mean_min, mean_max, std_min, std_max])))

        self.filter = A.transforms.AdditiveNoise(
            noise_type='gaussian', p=1.0, spatial_mode='per_pixel', noise_params={'mean_range': (mean_min, mean_max), 'std_range': (std_min, std_max)})

        if self.temporal_mode == 'constant':
            self.mask = self.filter(image=np.zeros(self.shape, dtype=np.float32))['image']

    def get_params(self):
        return {s: self.__getattribute__(s) for s in ['temporal_mode', 'mean_min', 'mean_max', 'std_min', 'std_max']}

    @staticmethod
    def get_params_info():
        info = {
            'temporal_mode': {'type': str, 'range': ['constant', 'per_frame'], 'info': 'Mode of applying noise. contant - one for all frames, per_frame - individual noise for each frame', 'default': 'per_frame'},
            'mean_min': {'type': float, 'range': [-1, 1], 'info': 'Minimal deviation for mean', 'default': 0.0},
            'mean_max': {'type': float, 'range': [-1, 1], 'info': 'Maximum deviation for mean', 'default': 0.0},
            'std_min': {'type': float, 'range': [0, 1], 'info': 'Maximum deviation for std', 'default': 0.1},
            'std_max': {'type': float, 'range': [0, 1], 'info': 'Maximum deviation for std', 'default': 0.1},
        }
        return info
