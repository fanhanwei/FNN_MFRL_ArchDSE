#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    forked from https://github.com/jfpower/anfis-pytorch
    Add support for u-arch DSE by FAN, Hanwei 
    Acknowledgement: 
    Originally developed by @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
'''

import itertools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F


dtype = torch.float


class FuzzifyVariable(torch.nn.Module):
    '''
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    '''
    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):  # No MF names supplied
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        '''Return the actual number of MFs (ignoring any padding)'''
        return len(self.mfdefs)

    def members(self):
        '''
            Return an iterator over this variables's membership functions.
            Yields tuples of the form (mf-name, MembFunc-object)
        '''
        return self.mfdefs.items()

    def pad_to(self, new_size):
        '''
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        '''
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        '''
            Yield a list of (mf-name, fuzzy values) for these input values.
        '''
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield(mfname, yvals)

    def forward(self, x):
        '''
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        '''
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    '''
    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        '''Return the number of input variables'''
        return len(self.varmfs)

    @property
    def max_mfs(self):
        ''' Return the max number of MFs in any variable'''
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        '''
            Print the variables, MFS and their parameters (for info only)
        '''
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                         mfdef.__class__.__name__,
                         ', '.join(['{}={}'.format(n, p.item())
                                   for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)

    def forward(self, x):
        ''' Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        '''
        assert x.shape[1] == self.num_in,\
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i+1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


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
        ''' Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        '''
        # Expand (repeat) the rule indices to equal the batch size:
        # batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        # Then use these indices to populate the rule-antecedents
        # ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        # ants.shape is n_cases * n_rules * n_in
        # Last, take the AND (= product) for each rule-antecedent
        # rules = torch.prod(ants, dim=2)
        
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
    '''
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    '''
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

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}'\
            .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        '''
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Can't have value 0 for weights, or LSE won't work:
        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use gels to do LSE, then pick out the solution rows:
        try:
            coeff_2d, _ = torch.gels(y_actual_2d, weighted_x_2d)
        except RuntimeError as e:
            print('Internal error in gels', e)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1]+1, -1)\
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

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


class PlainConsequentLayer(ConsequentLayer):
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''
    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        '''
        '''
        assert False,\
            'Not hybrid learning: I\'m using BP to learn coefficients'


class AnfisNet(torch.nn.Module):
    '''
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
    '''
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

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        '''
            Return an list of the names of the system's output variables.
        '''
        return self.outvarnames
    
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
                rstr = 'Rule {:2d}: IF {}'.format(i, rule_ants[i]) + ' THEN {} can increase'.format(self.outvarnames[selected])#list(map(lambda x,y: {x:round(y[0], 2)}, self.outvarnames, crow.tolist()))
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
    
    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        # y_pred = self.layer['weighted_sum'](self.weights, self.rule_tsk)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


# These hooks are handy for debugging:

def module_hook(label):
    ''' Use this module hook like this:
        m = AnfisNet()
        m.layer.fuzzify.register_backward_hook(module_hook('fuzzify'))
        m.layer.consequent.register_backward_hook(modul_hook('consequent'))
    '''
    return (lambda module, grad_input, grad_output:
            print('BP for module', label,
                  'with out grad:', grad_output,
                  'and in grad:', grad_input))


def tensor_hook(label):
    '''
        If you want something more fine-graned, attach this to a tensor.
    '''
    return (lambda grad:
            print('BP for', label, 'with grad:', grad))
