from tools import *
from .abstract import Abstract


class Stupid(Abstract):
    """
        Adds constant shift to each color
    """

    def __init__(self):
        super().__init__()

    def set_params(self, a, b, c, **params):
        self.a = a
        self.b = b
        self.c = c

    def get_params(self):
        return {s: self.__getattribute__(s) for s in 'abc'}

    def get_params_info(self):
        info = {
            'a': (int, [-125, 125], "Added to the first (red) chennel"),
            'b': (int, [-125, 125], "Added to the second (green) chennel"),
            'c': (int, [-125, 125], "Added to the third (blue) chennel"),
        }
        return info

    def apply_filter(self, frame):
        frame2 = frame.copy()
        frame2[..., 0] += self.a
        frame2[..., 1] += self.b
        frame2[..., 2] += self.c
        return frame2

    def get_objective(self, trial, **rest):
        return super().get_objective(trial, **rest)
