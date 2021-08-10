# cognitive-aether

## Python framework for development of classical and advanced Spectrum Sensing solutions in Cognitive Radio Networks.

This framework aims to offer an environment to generate OFDM data, simulate its transmition over different channels (AWGN, Rayleigh Flat/Frequency Selectiv Fading, Rice Flat/Frequency Selectiv Fading) and receive it with corresponding distortions based on which experiments can be performed. The framework is oriented towards research in fields of Spectrum Sensing and Cognitive Radio, for which focus is on the development of a consistent baseline of tools (Energy Detection algorithm, Machine Learning).
All processed signals (Gray code, QAM, OFDM TX, OFDM RX, noise, convolved, Bayes Shrink RX denoised, Visu Shrking RX denoised) can be saved in pandas DataFrames and CSV files. Furthermore, this framework aims to support development of Machine Learning / Deep Learning solutions through the posibility of synthesizing datasets with a wide spectrum of channel variations and signal distortions in controlled conditions, thus one can explore perfectly balanced or unbalanced datasets, specific scenarious and contexts.

## A soft reimplementation of Mathworks Rayleigh & Rician Flat & Frequency Selective Fading Channel
The framweork benefits of FadingChannel class which implements the same algorithm (Generalized Method for Exact Doppler Spread - GMEDS) as FadingChannel in Matlab - Communication Toolbox. Its usage is similar to the usage in Matlab with most of the tunable parameters of the blocks, making it possible to simulate and synthesize data from a wide spectrum of scenarious regarding radio channel behaviours.

NOTE: The development of this framework is in progress and more features are about to come.


![graymaptable](https://user-images.githubusercontent.com/53537308/111678680-3b02ad80-8829-11eb-9a2a-a2567d6fae51.png)


![6_snrs](https://user-images.githubusercontent.com/53537308/128870952-d6600b23-77f7-422d-aca4-c86f1fcafc39.png)

![5db_denoising](https://user-images.githubusercontent.com/53537308/128870975-7a646a7a-2aee-4ff9-ac83-0c9d8ab41bd0.png)

![psd_ray_flat_10db](https://user-images.githubusercontent.com/53537308/128871014-e7fa8a07-6668-4ec2-a6bc-7f9600eafc00.png)

![psd_ray_fselect_10db](https://user-images.githubusercontent.com/53537308/128871021-774a126a-0f36-4355-977c-a841666d9e87.png)


![ALL_SCENARIOS_ROCS_CH2](https://user-images.githubusercontent.com/53537308/128871160-61865af1-b290-470b-93b2-12a32d734f2e.png)


