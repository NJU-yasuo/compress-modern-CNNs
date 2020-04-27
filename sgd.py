import torch
import torch.optim as optim
import math

class SGD(optim.SGD):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
  
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for idx, p in enumerate(group['params']):
               
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                mask = torch.ne(p.data, 0).to(torch.float32)
                d_p =  d_p * mask 
                p.data.add_(-group['lr'], d_p)


        return loss


class SGD_caffe(optim.SGD):


    def step(self,closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
  
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for idx, p in enumerate(group['params']):
               
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)

                diff = group['lr'] * d_p + momentum * param_state['momentum_buffer']
                p.data.add_(-diff)                
                param_state['momentum_buffer'] = diff


        return loss
        
class CaffeSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super(CaffeSGD, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    if d_p.dim() != 1:
                        d_p.add_(weight_decay, p.data)
                if d_p.dim() == 1:
                    d_p.mul_( group['lr'] * 2 )
                else:
                    d_p.mul_(group['lr'])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                mask = torch.ne(p.data, 0).to(torch.float32)
                d_p =  d_p * mask 
                p.data.sub_(d_p) 

        return loss