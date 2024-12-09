from abc import ABC, abstractmethod


class Abstract(ABC):
    def __init__(self, shape):
        self.shape = shape
        self.params = {}

    @abstractmethod
    def set_params(self, **params):
        '''
        Sets parameters for the filter.

        Args:
            filter: filter that is going to be applied to the rgb image
            params: dictionary to change the parameters.
        '''
        for key, value in params.items():
            self.__setattr__(key, value)
        return

    @abstractmethod
    def get_params(self):
        '''
        Gets parameters for the filter.

        Returns:
            params: dictionary of the parameters.
        '''
        return self.params

    @abstractmethod
    def apply_filter(self, video_stream):
        """
        Processes the given video
        """
        pass

    @abstractmethod
    def get_objective(self, trial, **rest):
        """
        runs optuna

        returns: best trial
        """
        pass

    @abstractmethod
    def get_params_info():
        """
        Gets the parameters information and their ranges associated with the video processing.

        Returns:
            ParamsAndRange: dict, where keys are names of parameters and values are tuples of type, range of parameter and info.
        """
        pass
