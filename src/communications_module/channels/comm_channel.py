import numpy as np


class ChannelModel:

    def __init__(self, total_carriers_over_ch, channel_type='slow_fading'):
        self.total_carriers_ch = total_carriers_over_ch
        self.channel_type = channel_type
        self.ch_response = self._init_channel()

    def _init_channel(self):

        def scale(x, out_range=(0, 1), axis=None):
            domain = np.min(x, axis), np.max(x, axis)
            y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
            return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

        def get_channel_by_type():
            ch_fading_dict = {
                'slow_fading': self.slow_fading_ch_response()
            }
            return ch_fading_dict[self.channel_type]

        return scale(get_channel_by_type())

    def slow_fading_ch_response(self):
        arr = []
        arr_len = int(np.random.uniform(3, 10))
        for coef in range(arr_len + 1):
            ignore_flag = np.random.randint(low=0, high=2, size=1)[0]
            re, im = np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0)
            w = re + complex(im)
            if ignore_flag:
                arr.append(re)
            else:
                arr.append(w)
        print(arr)
        channel_response = np.array(arr)  # the impulse response of the wireless channel
        return np.fft.fft(channel_response, self.total_carriers_ch)
