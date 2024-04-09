#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
dtype = torch.float

class AntecedentLayer(torch.nn.Module):
    '''
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    '''
    def __init__(self, varlist, combinations):
        super(AntecedentLayer, self).__init__()
        # Count the (actual) mfs for each variable:
        mf_count = [var.num_mfs for var in varlist]
        # Now make the MF indices for each rule:
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))
        # mf_indices.shape is n_rules * n_in
        self.combinations = combinations
        sub_memFuncsByVariable = []
        for n in range(len(combinations)):
            sub_memFuncsByVariable.append([mf_count[i] for i in combinations[n]])
        self.sub_rules = []
        for i in range(len(combinations)):
            self.sub_rules.append(torch.tensor(list(itertools.product(*[range(n) for n in sub_memFuncsByVariable[i]]))))

    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        # mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        # for rule_idx in itertools.product(*[range(n) for n in mf_count]): #fully connected mode
        for subset, var_idxes in zip(self.sub_rules, self.combinations):
            for rule_idx in subset:
                thisrule = []
                for (varname, fv), i in zip([list(varlist.items())[i] for i in var_idxes], rule_idx):#varlist.items() #fully connected mode
                    thisrule.append('{} is {}'
                                    .format(varname, list(fv.mfdefs.keys())[i]))
                row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        
        indices = []
        for subset in self.sub_rules:
            indices.append(subset.expand((x.shape[0], -1, -1)))
            
        antecedents = []
        for comb, idx in zip(self.combinations, indices):
            antecedents.append(torch.gather(x.index_select(1,torch.tensor(comb)).transpose(1, 2), 1, idx))
            
        rules_sets = []
        for ant in antecedents:
            rules_sets.append(torch.prod(ant, dim=2))
            
        rules = torch.cat(rules_sets, 1) 
        return rules

class ConsequentLayer(torch.nn.Module):

    def __init__(self, d_in, d_rule, d_out, C=None):
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, 1])
        if C is not None:
            assert C.shape == c_shape, \
                'Coeff shape should be {}, but is actually {}'\
                .format(c_shape, C.shape)
            self._coeff = C
        else:
            self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * 1
                  y.shape: n_cases * n_out * n_rules
        '''
        # Use 1 for the constant term:
        x_plus = torch.ones(x.shape[0], 1) # simplified version of TSK, only a constant term as output
        # Need to switch dimansion for the multipy, then switch back:
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)  # swaps cases and rules


class AnfisNet(torch.nn.Module):

    def __init__(self, description, invardefs, outvarnames, combinations, C=None, hybrid=True):
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.combinations = combinations
        self.hybrid = hybrid
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_rules = sum([np.product([mfdefs[i].num_mfs for i in group]) for group in combinations])
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out, C)
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs, self.combinations)),
            # normalisation layer is just implemented as a function.
            ('consequent', cl),
            # weighted-sum layer is just implemented as a function.
            ]))
    
    def show_antecedents(self):
        return self.layer['fuzzify'].__repr__()

    def show_consequents(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' '*9+'THEN {}'.format(list(map(lambda x,y: {x:round(y[0], 2)}, self.outvarnames, crow.tolist()))))#crow.tolist()
        return '\n'.join(rstr)

    def simplified_rules(self, threshold=0.5):
        rule_base = []
        rule_menu = {}
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            if sum(crow) > threshold:
                selected = np.argmax(crow.tolist())
                rstr = 'IF {}'.format(rule_ants[i]) + ' THEN {} can increase'.format(self.outvarnames[selected])#list(map(lambda x,y: {x:round(y[0], 2)}, self.outvarnames, crow.tolist()))
                rule_base.append(rstr)
                
        for param in self.outvarnames:
            rule_menu[param] = []
            for rule in rule_base:
                if param+' can increase' in rule:
                    rule_menu[param].append(rule)
        return rule_base, rule_menu
    
    def find_rules(self):
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        new_coeff = []
        for i, crow in enumerate(self.layer['consequent'].coeff):
            if 'ROB is low' in rule_ants[i]: 
                temp = crow.tolist()
                temp[-5] = [1.]
                new_coeff.append(temp)
            else:
                new_coeff.append(crow.tolist())
        return torch.tensor(new_coeff, dtype=dtype, requires_grad=True)
    
