# AgileNuSrc
### Search for Gamma-Ray Counterparts of IceCube Neutrino IC170922A in the AGILE Public Archive.

The search for gamma ray counterparts of IceCube neutrino events is relevant for understanding the role of blazars as possible sources of cosmic neutrinos.
A new analysis of the IceCube neutrino event IC170922A region is performed in the time interval from January, 31 to March, 15 2018 using the data in the AGILE gamma-ray satellite public archive.
I present the candidate sources in the regions centred on the detected neutrinos and their light curves, providing estimates of the gamma ray flux above 100 MeV for the AGILE detections.
I use the Spectral Energy Distributions of the candidate source as input for the Bidirectional LSTM neural and obtain a probability for the source to be of blazar type.

### Requirements:
#### - To use the jupyter notebook with the analysis of the IceCube neutrino region
Install the following tools:
* Agilepy package (for installation visit https://agilepy.readthedocs.io/en/1.5.1/quickstart/installation.html)

#### - To use the Bi-LSTM neural network in order to classify the candidate neutrino source eventually found
Set a virtual environment with:
* Python v3.7
* Tensorflow v2.8
* Keras v2.8
* Numpy v1.21
* Pandas v1.3
* Matplotlib v3.2
* Scikit-learn v1.0
* Scipy v1.7

#### - To run the script for the Sed Builder online tool:
* Selenium

[![Build Status](https://app.travis-ci.com/elenagasparri/agilenusrc.svg?branch=main)](https://app.travis-ci.com/elenagasparri/agilenusrc)
