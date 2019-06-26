# A Pipeline for Entity Recognition and Relation Extraction Written in Pytorch

Main Requirements
-----
* python 3.6
* pytorch 1.0
* bioc 1.0
* nltk 3.3
* numpy 1.15
* pandas 0.24

Models
-----
Entity Recognition: BiLSTM-CRF

Relation Extraction: BiLSTM-Attention

Usage
-----
1. Train the pipeline

  ```
  python main.py -whattodo 1 -config default.config -output ./output -train_dir ./sample -dev_dir ./sample
  ```

  * whattodo=1: train ner and re models
  * config: configuration file
  * output: directory of saved models
  * train_dir: directory of training data
  * dev_dir: directory of development data

2. Extracting entities and relations using existing models

  ```
  python main.py -whattodo 2 -config default.config -output ./output -input ./input -predict ./predict
  ```

  * whattodo=2: use existing models to extract entities and relations from raw text
  * config: configuration file
  * output: directory of saved models
  * input: directory of raw text
  * predict: directory of predicted results in the bioc-xml format

3. Retraining the pipeline based on existing models

  ```
  python main.py -whattodo 1 -config default.config -output ./output -pretrained_model_dir ./pretrained -train_dir ./sample -dev_dir ./sample
  ```

  * whattodo=1: train ner and re models
  * config: configuration file
  * output: directory of saved models
  * pretrained_model_dir: directory of pretrained models, which are the models trained in Usage 1.
  * train_dir: directory of training data
  * dev_dir: directory of development data

Acknowledgement
-----
If you found the code is helpful, please cite:
```
@Article{info:doi/10.2196/12159,
author="Li, Fei and Liu, Weisong and Yu, Hong",
title="Extraction of Information Related to Adverse Drug Events from Electronic Health Record Notes: Design of an End-to-End Model Based on Deep Learning",
journal="JMIR Med Inform",
year="2018",
month="Nov",
day="26",
volume="6",
number="4",
pages="e12159",
issn="2291-9694",
doi="10.2196/12159",
url="http://medinform.jmir.org/2018/4/e12159/",
}
```

or

```
@article{li2017neural,
  title={A neural joint model for entity and relation extraction from biomedical text},
  author={Li, Fei and Zhang, Meishan and Fu, Guohong and Ji, Donghong},
  journal={BMC bioinformatics},
  volume={18},
  number={1},
  pages={198},
  year={2017},
  publisher={BioMed Central}
}
```

We mainly refered to the following work to write the code, so please also cite their work:
```
@inproceedings{yang2018ncrf,
 title={NCRF++: An Open-source Neural Sequence Labeling Toolkit},
 author={Yang, Jie and Zhang, Yue},
 booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
 Url = {http://aclweb.org/anthology/P18-4013},
 year={2018}
}
```

```
@InProceedings{N18-1111,
  author = 	"Chen, Xilun
		and Cardie, Claire",
  title = 	"Multinomial Adversarial Networks for Multi-Domain Text Classification",
  booktitle = 	"Proceedings of the 2018 Conference of the North American Chapter of the      Association for Computational Linguistics: Human Language Technologies,      Volume 1 (Long Papers)    ",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1226--1240",
  location = 	"New Orleans, Louisiana",
  doi = 	"10.18653/v1/N18-1111",
  url = 	"http://aclweb.org/anthology/N18-1111"
}
```
