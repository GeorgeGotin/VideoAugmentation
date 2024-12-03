from tools import *
from abstract import *


class MySpatter(VideoProcessor):
    def __init__(self):
        super().__init__()  # self.params = {}
        self.filter_function = albumentations.Spatter

    def set_params(self, params):
        '''
        Sets parameters for the filter.

        Args:
            filter: filter that is going to be applied to the rgb image
            params: dictionary to change the parameters.
        '''
        self.params = params
        self.params["p"] = 1         # we apply filter to all the frames
        # if we change it or mean the image will most likely either be yellow or black
        self.params["cutout_threshold"] = 0.68

        for tuple_param in ['mean', 'std', 'gauss_sigma', 'cutout_threshold', 'intensity']:
            if tuple_param in self.params:
                self.params[tuple_param] = (self.params[tuple_param], self.params[tuple_param])
        self.filter = self.filter_function(**self.params)

    def get_params(self):
        return self.params

    def apply_filter_video(self, input_path='videos/crowd_run_short_1920x1080_50.yuv', output_path=None, width=1920, height=1080, fps=50):
        """
        Processes the given video.
        may be in the abstract class but some filters are applied differently - may be changed in the future

        Args:
            input_path: the path to the yuv video
            output_path: the path where we save the filtered video
        """
        if output_path is None:
            output_path = 'videos/std={}intensity={}gauss_sigma={}.yuv'.format(self.params['std'][0],
                                                                               self.params['intensity'][0],
                                                                               self.params['gauss_sigma'][0])
        # Open the input YUV video
        with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
            while True:
                # Read and convert YUV frame to RGB
                s = read_yuv_frame(infile, width, height)
                if s is None:
                    print('done filtering')
                    break
                frame_rgb, (Y, U, V) = s

                # Apply filter to the RGB frame
                filtered_frame_rgb = self.filter(image=frame_rgb)['image']

                # Convert filtered RGB frame back to YUV
                filtered_frame_yuv = cv2.cvtColor(filtered_frame_rgb, cv2.COLOR_RGB2YUV)

                # Split Y, U, V channels from filtered YUV frame
                filtered_Y, filtered_U, filtered_V = cv2.split(filtered_frame_yuv)

                # Write the filtered YUV frame to the output file
                write_yuv_frame(outfile, filtered_Y, filtered_U, filtered_V)

        # Create PSNR result file path (same name as output YUV but with .txt extension)
        psnr_file = f"{os.path.splitext(output_path)[0]}_psnr.txt"

        # FFmpeg command to calculate PSNR and save to file
        # command = f"ffmpeg -f rawvideo -pix_fmt yuv420p -s {width}x{height} -i {input_path} -f rawvideo -pix_fmt yuv420p -s {width}x{height} -i {output_path} -lavfi psnr=\"stats_file={psnr_file}\" -f null -"
        tmp = "-s {}x{} -r {} -i".format(str(width), str(height), str(fps))
        command = f"ffmpeg {tmp} {output_path} {tmp} {
            input_path} -lavfi psnr=stats_file={psnr_file} -f null -"
        os.system(command)

        return parse_psnr_avg(psnr_file)

    def get_objective(self, trial, needed_psnr, input_path='videos/crowd_run_short_1920x1080_50.yuv',
                      width=1920, height=1080, fps=50):
        '''
        function of objective loss(returns objective loss for log regrression) with suggested parameters
        '''
        # mean = trial.suggest_float("mean", 0, 1, log=False, step=None)
        std = trial.suggest_float("std", 0, 1, log=False, step=None)
        intensity = trial.suggest_float("intensity", 0, 1, log=False, step=None)
        gauss_sigma = trial.suggest_float("gauss_sigma", 0, 20, log=False, step=0.5)
        # cutout_threshold = trial.suggest_float("cutout_threshold", 0, 1, log=False, step=None)

        # , "cutout_threshold":cutout_threshold}
        filter_params = {"std": std, "intensity": intensity, "gauss_sigma": gauss_sigma}
        self.set_params(filter_params)
        res = self.apply_filter_video(input_path, None, width, height, fps)
        output_path = 'videos/std={}intensity={}gauss_sigma={}.yuv'.format(self.params['std'][0],
                                                                           self.params['intensity'][0],
                                                                           self.params['gauss_sigma'][0])
        os.remove(output_path)
        if res is None:
            return 100
        score = np.abs(res - needed_psnr)
        print(res)
        return score

    def first_set(self, input_path='videos/crowd_run_short_1920x1080_50.yuv', psnr_file='new_psnr_spatter.txt',
                  output_path=None, width=1920, height=1080, fps=50):
        std_values = [0.3, 0.6, 0.9]
        intensity_values = [0.3, 0.6, 0.9]
        gauss_sigma_values = [2.5, 5, 7.5]

        with open(psnr_file, 'a') as ughhh:
            for std in std_values:
                for intensity in intensity_values:
                    for gauss_sigma in gauss_sigma_values:
                        print(std, intensity, gauss_sigma)
                        filter_params = {'p': 1, 'std': std,
                                         'intensity': intensity, 'gauss_sigma': gauss_sigma}
                        self.set_params(filter_params)
                        self.apply_filter_video(input_path, output_path, width, height, fps)
                        ughhh.write(
                            f"std={std}intensity={intensity}gauss_sigma={gauss_sigma}psnr={res}\n")

    def get_params_info(self, **idk_yet):
        message = '''Algorithm Info

This filter adds rain-like drops into every frame of the YUV-video
It takes the YUV-video converts it into an RGB-image applies the filter and converts it back into YUV-format.
The filter is based on Albumentations.Spatter filter, but the parameter choice was simplifiled significantly.

ParamInfo

Don't change parameters: p,  cutout_threshold, mean, mode
    p = 1, which means we apply our filter on all the frames of the video
    cutout threshold = 0.68, mean is set to standard, if we change either of those parameters it gets hard to 
       track if the frame gets black because it gets cut out, or if we set the cutout_threshold = 0 in some 
       implementations it may make all the videos look yellow-ish.
Toggled params:
    std: float [0, 1], in albumentations you get to change (std1, std2) so for each frame the param is randomly
        selected from range [std1, std2], but we are aiming to have consistent filter, therefore set
        std = std1 = std2
    gauss_sigma: float (0, +inf) - we changed it approximately in range (0, 10)
        the same idea about tuple simplification is implemented here
    intensity: [0, 1]
        the same idea about tuple simplification is implemented here
The rest of the parameters remained unchanged, therefore we don't guarantee the fiter would work with them changed.
'''
        print(message)
