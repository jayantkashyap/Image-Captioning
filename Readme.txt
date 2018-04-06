# Introduction
In this project we apply concepts of Natural Language Processing(NLP) and Deep Learning(DL) for the generation of the image description.
The project present a generative model based on a deep recurrent architecture that combines recent advances in computer vision and machine translation and that can be used to generate natural sentences describing an image.Here we use a neural and probabilistic framework to generate descriptions from images.
The model is to involve attention mechanism which would increase the efficiency of the system.
The model is trained and evaluated on Flickr8K and the result is captured and analysis is done.


## Dataset
Dataset can be requested and downloaded from this [link](https://forms.illinois.edu/sec/1713398). <br/>
However the dataset has been already downloaded and saved in 'Flickr8k' folder

After downloading, move the files to 'Flickr8k' folder and run the following commands<br>
`unzip Flickr8k_Dataset.zip`<br>
and<br>
`unzip Flickr8k_Text.zip`


## Installation
The application can be run in one of the two ways, either using Python Interpreter or using Jupyter Notebook.

**Python Environment**
This project require [Python 3.6](https://www.python.org/downloads/) interpreter.

To use the Python interpreter to run the project, first install the python packages being used in this project.

`pip3 install -r requirements.txt`<br/>
or
`pip install -r requirements.txt `

To run the application
`$python P3`

**Conda Environment**

The project requires [Anaconda 3](https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh)<br/>
To install Anaconda3 download the shell script from Anaconda website.

Run the following command
`bash Anaconda-latest-Linux-x86_64.sh`

Run the following to create environment and install packages
`conda env create -f project_env.yml`

Run the following to run the Jupyter Notebook
`jupyter notebook`

Select the '.ipynb' file.