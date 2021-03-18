# cognitive-aether

## Python framework for development of classical and advanced energy detection solutions in cognitive radio networks.

This framework offers an environment to generate OFDM data, transmit it over different channels (AWGN slow fading, Rayleigh, Rice) and receive it with channel distortions.
All processed signals (Gray code, OFDM TX, OFDM RX, noise, convolved, Bayes Shrink RX denoised, Visu Shrking RX denoised) can be saved in pandas DataFrames and CSV files. Furthermore, this framework aims to support development of Machine Learning / Deep Learning solutions through the posibility of synthesizing datasets with a wide spectrum of channel variations and signal distortions.

NOTE: The development of this framework is in progress and more features are about to come.


![graymaptable](https://user-images.githubusercontent.com/53537308/111678680-3b02ad80-8829-11eb-9a2a-a2567d6fae51.png)

Required signal power: 3.9613800508360746 [W]=[V^2]
Initial signal power: 9.161147906881277 [W]=[V^2]
Signal amplitude rescale factor: 0.6575795636029015 [Volts]
RX Signal power: 3.9614. Noise power: 1.9854, SNR [dB]: 3.0000
![TXRX_ofdm](https://user-images.githubusercontent.com/53537308/111678940-7dc48580-8829-11eb-8b3c-c965ab17d3b4.png)

![moredenoising](https://user-images.githubusercontent.com/53537308/111678734-481f9c80-8829-11eb-8844-7b3f18469aa1.png)

