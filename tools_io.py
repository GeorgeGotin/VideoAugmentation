import numpy as np
import cv2
import os

# Function to write YUV 4:2:0 frame


def write_yuv_frame(f, yuv_frame):
    """
    f - filestream, to which frame is written
    yuv_frame - np.array [H,W,C], frame if video
    """
    Y, U, V = yuv_frame.transpose(2, 0, 1)
    U_down = cv2.resize(U, (Y.shape[1] // 2, Y.shape[0] // 2))  # , interpolation=cv2.INTER_LINEAR)
    V_down = cv2.resize(V, (Y.shape[1] // 2, Y.shape[0] // 2))  # , interpolation=cv2.INTER_LINEAR)
    f.write(Y.tobytes())
    f.write(U_down.tobytes())
    f.write(V_down.tobytes())


class y4m_reader:
    """
    class for generators of frames
    """

    def __init__(self, path):
        """
        path - path to .y4m file
        return height, width of video
        """
        assert os.path.splitext(path)[-1] == ".y4m"
        assert os.path.exists(path)
        self.path = path
        input_file = open(path, 'rb')
        header = input_file.readline()
        self.header = header
        self.width = int(header.split(b'W')[1].split(b' ')[0])
        self.height = int(header.split(b'H')[1].split(b' ')[0])
        self.fps = str(header.split(b'F')[1].split(b' ')[0], 'utf-8')
        self.frame_size = self.width * self.height + (self.width // 2) * (self.height // 2) * 2
        self.shape = (self.height, self.width, 3)
        n_bytes = input_file.seek(0, 2)
        self.length = (n_bytes - len(header)) // (len('FRAME\n') + self.frame_size)
        input_file.close()

    def read_yuv_frame(self, f):
        """
        f - filestream, from which reading perfomance
        width - size, width of video frame
        height - size, height of video frame

        """
        yuv_frame = f.read(self.frame_size)
        if len(yuv_frame) < self.frame_size:
            return None

        # Separate Y, U, and V channels
        Y = np.frombuffer(yuv_frame[0:self.width*self.height],
                          dtype=np.uint8).reshape((self.height, self.width))
        U = np.frombuffer(yuv_frame[self.width*self.height:self.width*self.height + (self.width//2)
                                    * (self.height//2)], dtype=np.uint8).reshape((self.height//2, self.width//2))
        V = np.frombuffer(yuv_frame[self.width*self.height + (self.width//2)*(self.height//2):],
                          dtype=np.uint8).reshape((self.height//2, self.width//2))

        # Upsample U and V to match Y's resolution
        U_up = cv2.resize(U, (self.width, self.height))  # , interpolation=cv2.INTER_LINEAR)
        V_up = cv2.resize(V, (self.width, self.height))  # , interpolation=cv2.INTER_LINEAR)

        # Merge Y, U, V into a YUV image and convert to RGB
        yuv_img = cv2.merge([Y, U_up, V_up])
        rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)

        return (rgb_img / 255).astype(np.float32)

    def iter_frame(self):
        input_file = open(self.path, 'rb')
        input_file.readline()
        eof = False
        # counter = 0
        # rgb_sum = np.empty((n, self.height, self.width, 3), dtype=np.float32)
        while not eof:
            input_file.readline()
            rgb = self.read_yuv_frame(input_file)
            if rgb is None:
                eof = True
                break
            yield rgb

            # rgb_sum[counter] = rgb
            # counter += 1
            # if counter == n:
            #     counter = 0
            #     yield rgb_sum

        # yield rgb_sum
        input_file.close()
        return StopIteration

    def __iter__(self):
        return self.iter_frame()

    def __len__(self):
        return self.length


class y4m_writer:
    def __init__(self, path, width, height, fps):
        assert os.path.splitext(path)[-1] == ".y4m"
        self.output_file = open(path, 'wb')
        self.width = width
        self.height = height
        self.output_file.write(bytes(f'YUV4MPEG2 W{width} H{height} F{fps} Ip A1:1\n', 'utf-8'))

    def write_frame(self, frame):
        h, w, c = frame.shape
        assert h == self.height
        assert w == self.width
        self.output_file.write(bytes('FRAME \n', 'utf-8'))
        yuv_frame = cv2.cvtColor(np.clip(np.round(frame*255), 0,
                                 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
        write_yuv_frame(self.output_file, yuv_frame)

    def close(self):
        self.output_file.close()
