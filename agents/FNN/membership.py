#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from .anfis import AnfisNet

def _mk_param(val):
    '''Make a torch parameter from a scalar value'''
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))

class SigmoidMembFunc_const(torch.nn.Module):
    '''
        Sigmoid membership functions, defined by two parameters:
            b, the center
            c, the slope.
    '''
    def __init__(self, b, c):
        super(SigmoidMembFunc_const, self).__init__()
        self.b =b
        self.c =c

    def forward(self, x):
        val = torch.sigmoid(self.c * (x - self.b))
        # val = 1. / (1. + torch.exp(- self.c * (x - self.b))) # 1. / (1. + np.exp(- c * (x - b)))
        return val

    def pretty(self):
        return 'SigmoidMembFunc {} {}'.format(self.b, self.c)

class SigmoidMembFunc(torch.nn.Module):
    '''
        Sigmoid membership functions, defined by two parameters:
            b, the center
            c, the slope.
    '''
    def __init__(self, b, c):
        super(SigmoidMembFunc, self).__init__()
        self.register_parameter('b', _mk_param(b))
        # self.register_parameter('c', _mk_param(c))
        # self.b =b
        self.c =c
        self.b.register_hook(SigmoidMembFunc.b_log_hook)
        # self.c.register_hook(SigmoidMembFunc.c_log_hook)
    @staticmethod
    def b_log_hook(grad):
        '''
            Possibility of a log(0) or 0/0 in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        '''
        for item in grad[torch.isnan(grad)]:
            if item:
                print('nan in b of sigmoid')
        # grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        val = torch.sigmoid(self.c * (x - self.b))
        # val = 1. / (1. + torch.exp(- self.c * (x - self.b))) # 1. / (1. + np.exp(- c * (x - b)))
        return val

    def pretty(self):
        return 'SigmoidMembFunc {} {}'.format(self.b, self.c)

class BellMembFunc(torch.nn.Module):
    '''
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    '''
    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        # self.register_parameter('a', _mk_param(a))
        # self.register_parameter('b', _mk_param(b))
        # self.register_parameter('c', _mk_param(c))
        self.a =a
        self.b =b
        self.c =c
        
        # self.a.register_hook(BellMembFunc.a_log_hook)
        # self.b.register_hook(BellMembFunc.b_log_hook)
        # self.c.register_hook(BellMembFunc.c_log_hook)

    @staticmethod
    def a_log_hook(grad):
        for item in grad[torch.isnan(grad)]:
            if item:
                print('nan in a of bell')
        # grad[torch.isnan(grad)] = 1e-9
        return grad
    @staticmethod
    def b_log_hook(grad):
        '''
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        '''
        # for item in grad[torch.isnan(grad)]:
        #     if item:
        #         print('nan in b of bell')
        grad[torch.isnan(grad)] = 1e-9
        return grad
    @staticmethod
    def c_log_hook(grad):
        for item in grad[torch.isnan(grad)]:
            if item:
                print('nan in c of bell')
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def pretty(self):
        return 'BellMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_bell_mfs(a, b, clist):
    '''Return a list of bell mfs, same (a,b), list of centers'''
    return [BellMembFunc(a, b, c) for c in clist]



# Make the classes available via (controlled) reflection:
get_class_for = {n: globals()[n]
                 for n in ['BellMembFunc',
                           'SigmoidMembFunc',
                           'SigmoidMembFunc_const'
                           ]}

