from .stupid import Stupid
from .abstract import Abstract
from .spatter import Spatter
from .gaussian_blur import Gaussian_Blur
from .noise import UniformNoise, GaussianNoise


distortion_zoo = {
    'abstract': Abstract,
    'spatter': Spatter,
    'stupid': Stupid,
    'gaussian_blur': Gaussian_Blur,
    'uniform_noise': UniformNoise,
    'gaussian_noise': GaussianNoise,
}
