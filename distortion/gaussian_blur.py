from .abstract import Abstract
import albumentations as A


class Gaussian_Blur(Abstract):
    def __init__(self, shape):
        super().__init__(shape)
        self.ksize = 1
        self.sigma = 1
        self.filter = A.gaussian_blur

    def set_params(self, ksize, sigma, *args, **kwargs):
        self.kernel_size = ksize
        self.sigma = sigma

    def get_params(self):
        return {s: self.__getattribute__(s) for s in ['ksize', 'sigma']}

    def apply_filter(self, frame):
        frame = frame.copy()
        return self.filter(frame, ksize=self.ksize, sigma=self.sigma)

    def get_objective(self, trial, **rest):
        return super().get_objective(trial, **rest)

    @staticmethod
    def get_params_info():
        info = {
            'ksize': {'type': int, 'range': [1, float('inf'), 2], 'info': "kernel size of filter", 'default': 15},
            'sigma': {'type': float, 'range': [0, float('inf')], 'info': "sigma value of filter", 'default': 5},
        }
        return info