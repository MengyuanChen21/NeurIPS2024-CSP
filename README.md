## [NeurIPS 2024] Conjugated Semantic Pool Improves OOD Detection with Pre-trained Vision-Language Models

> **Authors**: Mengyuan Chen, Junyu Gao, Changsheng Xu.

> **Affiliations**: Institute of Automation, Chinese Academy of Sciences

### Paper
The Arxiv version is already available in this [link](https://arxiv.org/abs/2410.08611).

### Dependency
The project is based on MMClassification.
Please refer to [NegLabel](https://github.com/XueJiang16/NegLabel) for dependency installation.
Besides, we use python=3.8.18 and pytorch=1.10.2 and RTX 3090 GPUs.


### Data Preparation
We conduct main experiments with the validation set of ImageNet-1k LSVRC 2012 as ID data, and with iNaturalist, SUN, Places, and Textures as OOD data.

ImageNet-1k LSVRC 2012 can be downloaded from its [website](https://image-net.org/challenges/LSVRC/2012/index.php#).

Please follow instructions from the this [repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) to download the OOD datasets. Specifically, download iNaturalist, SUN, and Places with:
```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```
and download Textures from its original [website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).


**Note: The data paths in the configuration files need to be updated by users.**

To perform all experiments, users still need to prepare some other small datasets such as Stanford-Cars, which are introduced in the paper in details and easy to be downloaded.


### Get Started
Configuration files for all experiments are available in the `custom_config` folder.
To run the project, please follow:
```
PYTHONPATH=. python ./tools/ood_test.py custom_config/{config-file-name} foo
```
where `{config-file-name}` can be the name of any config file.

To reproduce the main results, just replace `{config-file-name}` with `clip-ViT-B16.py`.

### Acknowledgement
This project is built upon the repository of [NegLabel](https://github.com/XueJiang16/NegLabel) and [MOS](https://github.com/deeplearning-wisc/large_scale_ood). We sincerely thank their authors for their excellent work.

### Contact
Feel free to contact me (Mengyuan Chen: [chenmengyuan2021@ia.ac.cn](mailto:chenmengyuan2021@ia.ac.cn)) if anything is unclear or you are interested in potential collaboration.


