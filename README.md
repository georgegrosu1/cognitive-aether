# cognitive-aether

## Python framework for development of classical and advanced Spectrum Sensing solutions in Cognitive Radio Networks.

This framework aims to offer an environment to generate OFDM data, simulate its transmition over different channels (AWGN, Rayleigh Flat/Frequency Selectiv Fading, Rice Flat/Frequency Selectiv Fading) and receive it with corresponding distortions based on which experiments can be performed. The framework is oriented towards research in fields of Spectrum Sensing and Cognitive Radio, for which focus is on the development of a consistent baseline of tools (Energy Detection algorithm, Machine Learning).
All processed signals (Gray code, QAM, OFDM TX, OFDM RX, noise, convolved, Bayes Shrink RX denoised, Visu Shrking RX denoised) can be saved in pandas DataFrames and CSV files. Furthermore, this framework aims to support development of Machine Learning / Deep Learning solutions through the posibility of synthesizing datasets with a wide spectrum of channel variations and signal distortions in controlled conditions, thus one can explore perfectly balanced or unbalanced datasets, specific scenarious and contexts.

## A soft reimplementation of Mathworks Rayleigh & Rician Flat & Frequency Selective Fading Channel
The framweork benefits of FadingChannel class which implements the same algorithm (Generalized Method for Exact Doppler Spread - GMEDS) as FadingChannel in Matlab - Communication Toolbox. Its usage is similar to the usage in Matlab with most of the tunable parameters of the blocks.

NOTE: The development of this framework is in progress and more features are about to come.


![graymaptable](https://user-images.githubusercontent.com/53537308/111678680-3b02ad80-8829-11eb-9a2a-a2567d6fae51.png)

Required signal power: 3.9613800508360746 [W]=[V^2]
Initial signal power: 9.161147906881277 [W]=[V^2]
Signal amplitude rescale factor: 0.6575795636029015 [Volts]
RX Signal power: 3.9614. Noise power: 1.9854, SNR [dB]: 3.0000
![TXRX_ofdm](https://user-images.githubusercontent.com/53537308/111678940-7dc48580-8829-11eb-8b3c-c965ab17d3b4.png)

![moredenoising](https://user-images.githubusercontent.com/53537308/111678734-481f9c80-8829-11eb-8844-7b3f18469aa1.png)

