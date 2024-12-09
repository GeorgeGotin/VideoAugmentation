from tools import *
from .abstract import Abstract


class Stupid(Abstract):
    """
        Adds constant shift to each color
    """

    def __init__(self, shape):
        super().__init__(shape)

    def set_params(self, a, b, c, **params):
        self.a = a
        self.b = b
        self.c = c

    def get_params(self):
        return {s: self.__getattribute__(s) for s in 'abc'}

    @staticmethod
    def get_params_info():
        info = {
            'a': {'type': int, 'range': [-0.5, 0.5], 'info': "Added to the first (red,'default' : 0.0} chennel", 'default': 0.0},
            'b': {'type': int, 'range': [-0.5, 0.5], 'info': "Added to the second (green,'default' : 0.0} chennel", 'default': 0.0},
            'c': {'type': int, 'range': [-0.5, 0.5], 'info': "Added to the third (blue,'default' : 0.0} chennel", 'default': 0.0},
        }
        return info

    def apply_filter(self, frame):
        frame2 = frame.copy()
        frame2[..., 0] += self.a
        frame2[..., 1] += self.b
        frame2[..., 2] += self.c
        return frame2

    def get_objective(self, video_stream, needed_psnr):
        def func(trial):
            a = trial.suggest_float('a', -0.5, 0.5, log=False, step=None)
            b = trial.suggest_float('b', -0.5, 0.5, log=False, step=None)
            c = trial.suggest_float('c', -0.5, 0.5, log=False, step=None)
            self.set_params(a, b, c)

        return func
