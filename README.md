# Pytorch implementation of Sparse NN

### Experimental results

To be added.

### Quick start example
- start command
```
python main.py --arch=vgg16 --pretrained --finetune_lr=1e-3 --lr_decay=30 --prune --conf=conf/vgg.conf
python main.py --arch=alexnet --pretrained --finetune_lr=1e-3 --lr_decay=30 --prune --conf=conf/admm.conf
```
- conf is used to define the layers to be pruned, expected sparsity and other hyper-parameters for pruning.

### The pruning schemes follow these papers
```
@inproceedings{han2015learning,
  title={Learning both weights and connections for efficient neural network},
  author={Han, Song and Pool, Jeff and Tran, John and Dally, William},
  booktitle={Advances in neural information processing systems},
  pages={1135--1143},
  year={2015}
}

@article{zhang2018systematic,
  title={A systematic DNN weight pruning framework using alternating direction method of multipliers},
  author={Zhang, Tianyun and Ye, Shaokai and Zhang, Kaiqi and Tang, Jian and Wen, Wujie and Fardad, Makan and Wang, Yanzhi},
  journal={arXiv preprint arXiv:1804.03294},
  year={2018}
}
```
