import argparse
import os
import random
import shutil
import time
import warnings
import sys
import copy
import configparser

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
 

import models
from utils import *
from sgd import *
import sparse.admm_sparse as admm_sparse
from optimizer import PruneAdam

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/media/shared-corpus/ImageNet/', 
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--finetune_lr', '--finetune-learning-rate', default=1e-3, type=float,
                    metavar='FT-LR', help='initial learning rate for finetune', dest='finetune_lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False,
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--state_dict', default='', type=str, help='state_dict file to load')
parser.add_argument('--prune', dest='prune', action='store_true',
                    help='Whether to prune model.')
parser.add_argument('--conf', default='/home/ww/admm-res/conf/admm_res50.conf', type=str,
                    help='Scheme chosen for pruning. admm and naive mod are available.')
   

best_acc1 = 0

import warnings
warnings.filterwarnings('ignore')



def read_config(config_file):
    print ('read conf from: ', config_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    conf_dict = {}
    sparse_config = config.items('SparseConfig')
    for sc in sparse_config:
        conf_dict[sc[0]] = sc[1]
    return conf_dict


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    

    # create model
    if args.pretrained or args.state_dict:
        if args.state_dict:
            state_dict = torch.load(args.state_dict)
        else: 
            state_dict = None 
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
      

    #print_prune(model)

    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    

    # optionally resume from a checkpoint
    if args.resume:

        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
#            args.start_epoch = checkpoint['epoch']
#            best_acc1 = checkpoint['best_acc1']
            #model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint)
#            optimizer.load_state_dict(checkpoint['optimizer'])
            #model.load_state_dict(checkpoint)
#            print("=> loaded checkpoint '{}' (epoch {})"
#                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))    
            

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()        

 

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])


    train_dataset = datasets.ImageFolder(
             traindir,
             transforms.Compose([
                 transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize,
             ]))
         
   
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
         train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
         num_workers=args.workers, pin_memory=True, sampler=train_sampler,
         drop_last=False)




    val_loader = torch.utils.data.DataLoader(
             datasets.ImageFolder(valdir, transforms.Compose([
                 transforms.Resize(256), 
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),  
                 normalize,                                              
             ])),        
         batch_size=args.batch_size, shuffle=False,
         num_workers=args.workers, pin_memory=True)



    if args.evaluate:  
  
                
        print(model)
              
        validate(val_loader, model, criterion, args)
        
        
        return 
    
    if args.prune:
        conf = read_config(args.conf)
        sparse_param = admm_sparse.ADMMParameter(conf)
        sparse_param.InitParameter(model)
        regulier = nn.MSELoss(reduction='sum')
    
    
    #set up train folder
    current_time = time.strftime('-%m%d-%H%M', time.localtime(time.time()))
    current = 'checkpoints/' + args.arch + current_time + '/'
    os.mkdir(current)
    #set up hook
    
    if args.prune:
        sparse_param.CheckInformation(model)
 
    
    optimizer_admm = torch.optim.SGD(model.parameters(), args.lr,weight_decay=5e-4,momentum=0.9)        
  #  optimizer_admm = PruneAdam(model.named_parameters(),args.lr,weight_decay=5e-4)
          
    for _ in range(20):
      
        for epoch in range(12):

            optimizer_admm.param_groups[0]['lr'] = 1e-3
            do_admm = True
            if epoch >= 2:
                do_admm = True
                optimizer_admm.param_groups[0]['lr'] = 1e-4
            sparse_param = train(train_loader, model, criterion, optimizer_admm, epoch, args,sparse_param,regulier,do_admm,_)
            acc1 = validate(val_loader, model, criterion, args)
        
 
        sparse_param.UpdateParameter(model)
        sparse_param.CheckInformation(model)

    sparse_param._PruneModel(model)
    print_prune(model)  
    torch.save(model.cpu().state_dict(),'mobile_5x.pth')
    optimizer = SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=5e-4)  
    #optimizer = PruneAdam(model.named_parameters(),args.lr,weight_decay=5e-4)
     
    model.cuda()     

    for epoch in range(40):
       
        optimizer.param_groups[0]['lr'] = 1e-2
        if epoch >= 10:
            optimizer.param_groups[0]['lr'] = 1e-3
        if epoch >= 20:
            optimizer.param_groups[0]['lr'] = 1e-4
        if epoch >= 30:
            optimizer.param_groups[0]['lr'] = 1e-5
        train(train_loader, model, criterion, optimizer, epoch, args, None, None, False)
        acc1 = validate(val_loader, model, criterion, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
      
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, current=current)
    torch.save(model.cpu().state_dict(),'res_20x.pth')
     
def train(train_loader, model, criterion, optimizer, epoch, args, sparse_param = None, regulier = None, do_admm = False,admm_iter=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i,(input,target) in enumerate(train_loader):
     
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
      #  input = data[0]["data"].cuda(non_blocking=True)
      #  target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        optimizer.zero_grad()
        # compute output
        output = model(input)
        loss = criterion(output, target)        

        if do_admm:
     
            loss_admm = sparse_param.ComputeRegulier(regulier, model)
            admm_loss = loss_admm.item()
            loss += loss_admm
            
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        
        # compute gradient and do SGD step
      
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if do_admm:
            if i % 1000  == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Admm_loss{admm_loss:.10f} {n:}'.format(
                   epoch, i, 5005, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5,admm_loss=admm_loss,n=admm_iter))
        else:
            if i % 1000 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, 10000, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return sparse_param
 
    
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i,(input,target) in enumerate(val_loader):
          
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            
           # input = data[0]["data"].cuda(non_blocking=True)
           # target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
 
            
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
##
            if i % 1000 == 0:
                print('Test: [{0}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, batch_time=batch_time, loss=losses,
                       top1=top1 , top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))        
    return top1.avg


def save_checkpoint(state, is_best, current, filename='checkpoint.pth.tar'):
    checkpoint_filename = current + filename
    torch.save(state, checkpoint_filename)
    if is_best:
        shutil.copyfile(checkpoint_filename, current + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.pretrained or args.state_dict : 
        lr = args.finetune_lr * (0.9 ** (epoch // 2))
    else:    
        lr = args.lr * (0.1 ** (epoch // 30))
    print ('lr:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

 

 
if __name__ == '__main__':
    main()
