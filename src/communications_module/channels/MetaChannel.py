from abc import ABC


class MetaChannel(ABC):

    def __init__(self, channel_type: str):
        super(MetaChannel, self).__init__()
        self.channel_type = channel_type
        self.ch_response = None

    def _init_channel(self):
        pass
