# Pytorch implementation of Sparse NN

### Experimental results

x18 for res50 with 1.5% top-5 acc loss(imagenet)   
x69 for res18 with 0.6% top-1 acc loss(cifar10)

### Quick start example
- start command
```
python main_xx.py --arch=xx --pretrained --finetune_lr=1e-3 - --prune --conf=conf/vgg.conf
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
