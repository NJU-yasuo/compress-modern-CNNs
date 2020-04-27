from __future__ import division
import torch
import imp  
import numpy as np

class Swap(object):
    def __init__(self):
        NotImplemented
    def __call__(self, tensor):
 
        new_tensor = tensor.clone()
        new_tensor[0,:,:] = tensor[2,:,:]
        new_tensor[2,:,:] = tensor[0,:,:]
        return new_tensor 


class Normalize(object):
    def __init__(self):
        NotImplemented
    def __call__(self, input):
        mean = np.load('mean.npy')
        input = input - mean.transpose(1,2,0)
        print(input)
        
        
        return input.astype(np.uint8)



def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and param.dim()!=1 :
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))
