# Lung Quant 

---

Lung Segmentation and Registration from Lung CTs using Deep Convolutional Neural Networks

Contact: Kyle Hasenstab (kahasenstab@sdsu.edu), Joseph Tabalon (jtabalon@sdsu.edu)

## Introduction 

---
 
This software provides functions to segment lobes from CT scans using a custom deep CNN model, deformably register an expiratory series to an inspiratory series, and compute lung CT measurements. 

If using this software influences your project, please cite the paper below:

[insert paper citation here]

This software includes tools to:

i) Preprocess the CT images to use with our Segmentation + Registration models

ii) Segment a pair of Lung CT Images

iii) Deformably register expiratory series to inspiratory series 

iv) Compute Lung CT measurements

v) Generate Attenuation Difference Map (ADM)


## Project Organization
```
├── LICENSE                 <- MIT License
│
├── README.md               <- The top-level README for developers using this project
│
│
├── models                  <- pretrained models: transformer, segmentation, registration
│
│
├── data                    <- Directory to input inspiratory/expiratory dicoms (default: None)
│   ├── insp                <- Directory for inspiratory series dicoms (default: None)
│   ├── exp         	    <- Directory for expiratory series dicoms (default: None)
│
│
│
├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
│                         	   generated with `pip freeze > requirements.txt`
│
├── lung_quant.py           <- Source code for use in this project.
```


## Requirements




---




- The required packages are in `requirements.txt`.


### It is recommended to use an Anaconda Environment
**TODO** complete requirements section
- ./install.sh
- pip install -r requirements.txt

## Run the Scripts

The user should choose to run the scripts in the native LungQuant directory. This will preserve the relative location of the pretrianed models needed to segment and register each pair of lungs.

You may find a total list of arguments you may specify during the initial run by running the following:

`python -m lung_quant.py --help`

### Prepare Data Directory
When cloning the repo, a `/data/` folder contains the `/insp/` and the `/exp/` directories, which is where a user should place their respective dicoms. 

Or, you may choose to specify arguments such as specific data directories and output directories during the running of the `LungQuant.py` script.

Example:

`python -m lung_quant.py -i ~/insp_dicoms/ -e ~/exp_dicoms/ -m ~/metrics/`

### Loading Models

The user should ensure the pretrained models (segmentation, registration, transformer) are all in a subdirectory, `/models/` below the location of the `lung_quant.py` file to ensure accurate loading of the models.

### Calculating and Exporting Metrics

By default, our algorithm will output metrics calculated from each pair of lungs, as well as metrics obtained after the registration. These metrics will be output to a `/metrics/` folder - created in the same directory as `lung_quant.py` if it is not already preasent. 

### Generate Attenuation Difference Map

By default, our algorithm will output calculated images of our Attenuation Difference Map, and a .gif of the concatenated difference maps. These images
will be found in a `/adm/` folder - created in the same directory located as `lung_quant.py` if it is not already present. You may also specify to not
generate these images by using a particular argument during the initial run.

