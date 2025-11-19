# 🌌 Chimera: A Cognitive & Aesthetic Sentiment Causality Understanding Framework

This repository provides the source code and datasets for our paper:  
📄 **[Exploring Cognitive and Aesthetic Causality for Multimodal Aspect-Based Sentiment Analysis](https://arxiv.org/pdf/2504.15848?)**

## 🚀 Overview

<p align="center">
  <img src="./chimera.png" alt="Chimera Framework" width="800"/>
</p>

Chimera is a unified framework designed to model **cognitive reasoning** and **aesthetic perception** in multimodal aspect-based sentiment analysis (MABSA). It integrates text–image interaction, sentiment causality understanding, and fine-grained multimodal reasoning.

## 📦 Environment Requirements

We recommend using **conda** for environment management:

```bash
conda env create -f Chimera.yaml
📚 Datasets
1. Constructed Datasets
Twitter2015 → data/twitter2015

Twitter2017 → data/twitter2017

Political Twitter → data/political_twitter

2. Image Feature Downloads
Image features can be downloaded from:

🔗 Google Drive
https://drive.google.com/drive/folders/1bLEz_sr1loC4TPa39S30L4a7lB2r-ACr?usp=sharing

Place them in the following locations:

For Twitter2015 → data/twitter2015

For Twitter2017 → data/twitter2017

For Political Twitter:

Move all contents from data/twitter2015 and data/twitter2017 into data/political_twitter

Extract the two .zip files into the same directory

🤖 Pretrained Model
Chimera uses Flan-T5 as the backbone.

Download the pretrained checkpoint:

📥 google/flan-t5-base

Save it under:

bash
Copy code
pretrained/flan-t5-base
🏋️ Training & Evaluation
Run the corresponding training script:

bash
Copy code
python run_chimera_15.py
python run_chimera_17.py
python run_chimera_political.py
📖 Citation
If you find this repository helpful, please consider starring ⭐ the repo and citing our papers:

graphql
Copy code
@article{xiao2025exploring,
  title={Exploring Cognitive and Aesthetic Causality for Multimodal Aspect-Based Sentiment Analysis},
  author={Xiao, Luwei and Mao, Rui and Zhao, Shuai and Lin, Qika and Jia, Yanhao and He, Liang and Cambria, Erik},
  journal={IEEE Transactions on Affective Computing},
  year={2025},
  publisher={IEEE}
}

@article{xiao2024atlantis,
  title={Atlantis: Aesthetic-oriented multiple granularities fusion network for joint multimodal aspect-based sentiment analysis},
  author={Xiao, Luwei and Wu, Xingjiao and Xu, Junjie and Li, Weijie and Jin, Cheng and He, Liang},
  journal={Information Fusion},
  volume={106},
  pages={102304},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{xiao2024vanessa,
  title={Vanessa: Visual connotation and aesthetic attributes understanding network for multimodal aspect-based sentiment analysis},
  author={Xiao, Luweи and Mao, Rui and Zhang, Xulang and He, Liang and Cambria, Erik},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={11486--11500},
  year={2024}
}
🙏 Acknowledgements
Chimera is built upon the excellent foundations of:

MDCA

LAPS

We extend our sincere gratitude to all contributors for their valuable insights and support.

yaml
Copy code
