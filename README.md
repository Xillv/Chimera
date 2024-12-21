# Chimera: A cognitive and aesthetic sentiment causality understanding framework

This repository contains the source code and datasets associated with the paper titled "Exploring Cognitive and Aesthetic Causality for Multimodal Aspect-Based Sentiment Analysis."


----------

## Overview

<p align="center">
  <img src="./chimera.pdf" width="800"/>
</p>

----------

### Requirements

- conda env create -f Chimera.yaml


----------

### Datasets
1. Constructed datasets: Twitter2015 ([twitter2015](data/twitter2015)), Twitter2017 ([twitter2017](data/twitter2017)) and Political Twitter ([political_twitter](data/political_twitter)).
2. Image features can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1bLEz_sr1loC4TPa39S30L4a7lB2r-ACr?usp=sharing). Place the downloaded files in the directories `data/twitter2015` and `data/twitter2017`, respectively. For the `political_twitter` dataset, move the contents of `data/twitter2015` and `data/twitter2017` into `data/political_twitter` and extract the two `.zip` files into the same directory.


----------

### Pretrained Models
- The Flan-T5 model is utilized as the backbone. Download the pre-trained model [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) and save it in the directory `pretrained/flan-t5-base`.


----------

###  Training and Evaluating 

python run_chimera_15.py

python run_chimera_17.py

python run_chimera_political.py

### Acknoeledgement

This work is primarily built upon the repositories of MDCA and LAPS. Sincere gratitude is extended to everyone who contributed to this project for their invaluable support and dedication.
