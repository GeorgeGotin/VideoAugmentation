from abc import ABC, abstractmethod

class VideoProcessor(ABC):        
    def __init__(self):
        self.filter = None
        self.params = {}
        
    @abstractmethod
    def set_params(self, filter, params):
        '''
        Sets parameters for the filter.

        Args:
            filter: filter that is going to be applied to the rgb image
            params: dictionary to change the parameters.
        '''
        #self.filter = filter(**params) - sometimes won't work
        #self.params = params
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
    def apply_filter_video(self, **rest):
        """
        Processes the given video.
        """
        pass

    @abstractmethod
    def get_objective(self, **rest):
        """
        runs optuna

        returns: best trial
        """
        pass

    @abstractmethod
    def first_set(self, **rest):
        """
        runs filter on grid of parameters
        """
        pass
        
    @abstractmethod
    def get_params_info(self, **idk_yet):
        """
        Gets the parameters information and their ranges associated with the video processing.

        Returns:
            ParamsAndRange: The parameters and range.
        """
        pass
