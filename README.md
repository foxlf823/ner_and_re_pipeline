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
  python main.py -whattodo 1 -config default.config -output ./output
  ```

  * whattodo=1: train ner and re models
  * config: configuration file
  * output: directory of saved models

  The paths of training and development datasets are set in the configuration file.

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
  python main.py -whattodo 1 -config default.config -output ./output -pretrained_model_dir ./pretrained
  ```

  * whattodo=1: train ner and re models
  * config: configuration file
  * output: directory of saved models
  * pretrained_model_dir: directory of pretrained models, which are the models trained in Usage 1.

Acknowledgement
-----
If you found the code is helpful, please cite:
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
