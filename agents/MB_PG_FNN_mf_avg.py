import os 
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import stats
from random import choice
from copy import deepcopy
import numpy as np

import torch
import torch.distributions as tdist
from analytic.Analytic_model_torch import CPI, CacheMissrate, area_regression, McPAT
from hf.vcs import get_cpi_vcs
from agents.FNN import anfis
from agents.FNN.membership import SigmoidMembFunc, BellMembFunc, SigmoidMembFunc_const
from collections import OrderedDict

from util.common import record

    
class Agent:
    def __init__(self, params, algs, args):
        self.num_episode = args.num_episode
        self.num_step = args.num_step
        self.lr = args.lr
        self.system = self.set_system(params)
        self.params = params
        self.algs = algs
        self.init_state = params.clone().detach()
        self.area_limit = args.area_limit
        self.warmup = args.warmup
        self.low_fidelity_iter = args.lf_iter
        self.sigma = args.sigma
        self.hf_sigma = args.hf_sigma
        self.decay_rate = args.decay_rate
        self.hf_decay_rate = args.hf_decay_rate
        self.lf_cpi_pool = []
        self.hf_design_pool = []
        self.lf_best_pool = []
        self.recent_cpi = [0, 0, 0, 0, 0]
        self.best_lf_cpi = 20 # random number larger than largest possible result
        self.best_hf_cpi = 20
        self.baseline_lf_cpi = 5#10.48
        self.baseline_hf_cpi = None
        self.opt = args.opt
        self.log_path = args.log_path
        self.record = record(self.log_path, params, args)
        # self.benchmark = "average"
        self.args = args
        self.hf_data_paths = []
        self.high_fidelity_dicts = []
        self.benchmarks = [alg['alg_name'] for alg in self.algs]
        self.incremental = True # args.incremental
        for idx, benchmark in enumerate(self.benchmarks):
            self.hf_data_paths.append(os.path.join(args.home_dir, 'dataset/vcs/{}.txt'.format(benchmark)))
            with open(self.hf_data_paths[idx], "r") as r:
                sim_data = r.readlines()
            _dict = {}
            for item in sim_data:
                pair = item.strip().split('--')
                cpi = float(pair[1])
                _dict[pair[0][1:-1]] = cpi
            self.high_fidelity_dicts.append(_dict)

        print(args)
        invardefs = [
            ('cycle',  OrderedDict(zip(['low', 'medium', 'high'], [SigmoidMembFunc_const(0.8, -10),  BellMembFunc(1, 3, 1.5),  SigmoidMembFunc_const(3, 10)]     ))),
            ('area',   OrderedDict(zip(['low', 'medium', 'high'], [SigmoidMembFunc_const(5, -10),    BellMembFunc(1, 3, 6),    SigmoidMembFunc_const(7, 10)]       ))),
            ('l1',  OrderedDict(zip(['low', 'high'], [SigmoidMembFunc(8, -10), SigmoidMembFunc(8, 10)]     ))),
            ('l2',  OrderedDict(zip(['low', 'high'], [SigmoidMembFunc(11, -10), SigmoidMembFunc(11, 10)]     ))),
            ('ROB',    OrderedDict(zip(['low', 'high'],  [SigmoidMembFunc(3, -10),  SigmoidMembFunc(3, 10)]     ))),
            ('IQ',    OrderedDict(zip(['low', 'high'],  [SigmoidMembFunc(3, -10),  SigmoidMembFunc(3, 10)]     ))),
            ('FU',    OrderedDict(zip(['low', 'high'],  [SigmoidMembFunc(5, -10),  SigmoidMembFunc(5, 10)]     ))),
            ('decode',    OrderedDict(zip(['low', 'high'],  [SigmoidMembFunc(3, -10),  SigmoidMembFunc(3, 10)]     ))),
            ]
        outvars = ['l1set', 'l1way', 'l2set', 'l2way', 'mshr', 'decode', 'ROB', 'int', 'float', 'mem', 'IQ']
        combinations = [[0,1,2,3,4,5,6,7]]
        # manual_init = torch.full(torch.Size([288, len(outvars), 1]), 0.5, dtype=torch.float, requires_grad=True)
        self.model = anfis.AnfisNet('DSE', invardefs, outvars, combinations, C=None, hybrid=False)
    
    def current_params(self):
        params = [self.system["L1sets"], self.system["L1ways"], self.system["L2sets"], self.system["L2ways"], 
                    self.system["mshr"], self.system["dispatch"], self.system["ROB"], 
                    self.system["FUint"], self.system["FUfp"], self.system["FUmem"], self.system["IQ"]]
        return [int(x) for x in params]
    
    def show_params(self):
        return "L1: {} {} ".format(int(self.system["L1sets"]), int(self.system["L1ways"])) + "L2: {} {} ".format(int(self.system["L2sets"]), int(self.system["L2ways"])) + \
                "mshr: {} ".format(int(self.system["mshr"])) + "decode: {} ".format(int(self.system["dispatch"])) + "ROB:{} ".format(int(self.system["ROB"])) + \
                    "FU: {} {} {} ".format(int(self.system["FUint"]), int(self.system["FUfp"]), int(self.system["FUmem"])) + "IQ:{} ".format(int(self.system["IQ"]))
    
    def arch_update(self, updates):
        def bound(param, lower, upper): 
            flag = True
            if param >= upper:
                flag = False
                param = upper
            return max(param, lower), flag
        bound_list = [
            bound(float(self.system["L1sets"]) * 2**updates[0], 16, 64),
            bound(float(self.system["L1ways"]) * 2**updates[1], 1, 16),
            bound(float(self.system["L2sets"]) * 2**updates[2], 128, 2048),
            bound(float(self.system["L2ways"]) * 2**updates[3], 2, 16),
            bound(float(self.system["mshr"]) * 2*updates[4], 2, 8),
            bound(float(self.system["dispatch"]) + updates[5], 1, 5),
            bound(float(self.system["ROB"]) + 32*updates[6], 32, 160),
            bound(float(self.system["FUint"]) + updates[7], 1, 5),
            bound(float(self.system["FUfp"]) + updates[8], 1, 5),
            bound(float(self.system["FUmem"]) + updates[9], 1, 2),
            bound(float(self.system["IQ"]) * 2**updates[10], 2, 24)
            ]
        not_max = [item[1] for item in bound_list]
        # max_FU = max(bound_list[7], bound_list[8], bound_list[9])
        # bound_list[5] = max(bound_list[5], max_FU) # decode width should be larger than FU issue width
        # if bound_list[5][0] == 5:
        #     bound_list[10] = max(bound_list[10], (8., True))
        # elif bound_list[5][0] >= 3:
        #     bound_list[10] = max(bound_list[10], (4., True))
        # else:
        #     pass
        self.params = torch.tensor([item[0] for item in bound_list], requires_grad=True, dtype=torch.float32)
        self.system = self.set_system(self.params)
        with open(os.path.join(self.log_path, "details.txt"),"a") as f:
            f.write("update: {}\n".format([int(x) for x in updates])) 
        return not_max
    
    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)    
    
    def evaluate(self): # model with gradients
        cpi_total = 0
        miss1_total = 0
        miss2_total = 0
        for alg in self.algs:
            cpi_total += CPI(self.system, alg)
            miss1_total += CacheMissrate(1, self.system, alg)
            miss2_total += CacheMissrate(2, self.system, alg)
        cpi_avg = cpi_total / len(self.algs)
        miss1_avg = miss1_total / len(self.algs)
        miss2_avg = miss2_total / len(self.algs)
        with open(os.path.join(self.log_path, "details.txt"),"a") as f:
            f.write('params: {}\ncpi: {:.3f}, miss1: {:.2f}, miss2: {:.2f}, '.format(self.current_params(), float(cpi_avg), float(miss1_avg), float(miss2_avg)))
        area = McPAT(self.system, self.algs[0], cpi_avg, miss1_avg, miss2_avg) # alg cpi missrate have no effect to the area of mcpat
        # area = area_regression(self.system)
        # area = self.area_model(torch.Tensor(self.area_input()))
        with open(os.path.join(self.log_path, "details.txt"),"a") as f:
            f.write('area: {:.4}\n'.format(float(area)))#['%.2f'% x for x in metrics]
        return [cpi_avg, miss1_avg, miss2_avg, area]

    def get_input(self, cpi, area):
        l1size = int(torch.log2(self.params[0]*self.params[1])+self.params[4])
        l2size = int(torch.log2(self.params[2]*self.params[3]))
        # IS = int(self.params[6]//32 + torch.round(torch.log2(self.params[10])))
        ROB = int(self.params[6]//32)
        IQ = int(torch.round(torch.log2(self.params[10])))
        FU = int(self.params[7]+self.params[8]+self.params[9])
        decode = int(self.params[5])
        return torch.tensor([[float(cpi), float(area), l1size, l2size, ROB, IQ, FU, decode]]) #torch.tensor([[float(cpi), float(area), l1size, l2size, IS, FU, decode]])
    
    def high_fidelity_evaluate(self, params):
        cpi_total = 0
        for benchmark in self.benchmarks:
            cpi = get_cpi_vcs(params, benchmark, self.log_path)
            cpi_total += cpi
        cpi_avg = cpi_total / len(self.benchmarks)
        return cpi_avg
    
    def optimize(self):
        if self.opt =='adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt =='rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.5)
        else:
            RuntimeError('Current optimizer {} is not support, please specify a valid optimizer'.format(self.opt))
        print("Start training")
        with open(os.path.join(self.log_path, "details.txt"),"a") as f: f.write("Start training\n") 
        for episode in range(1, self.num_episode+1):
            print('\n--- episode: {} ---\n'.format(str(episode)))
            with open(os.path.join(self.log_path, "details.txt"),"a") as f: f.write('\n\n--- episode: {} ---\n'.format(str(episode))) 
            loss_pool = []
            step_reward_pool = []
            log_prob_pool = []
            self.optimizer.zero_grad()
            with open(os.path.join(self.log_path, "details.txt"),"a") as f: f.write('Starting point:') 
            metrics = self.evaluate()
            cpi = metrics[0]
            area = metrics[3]
            last_cpi = float(cpi)
            last_area = float(area)
            not_max = [True for n in range(len(self.params))]
            run_line_search = False
            for step in range(1, self.num_step+1):# self.args.max_iter+1
                with open(os.path.join(self.log_path, "details.txt"),"a") as f: f.write('--iter:{}\n'.format(step))
                self.step = step
                cpi.backward()
                updates = [0 for n in range(len(self.params))]
                y_pred = self.model(self.get_input(metrics[0], area))
                grads = self.params.grad.data.tolist()
                recommand = []
                # with open(os.path.join(self.log_path, "details.txt"),"a") as f: f.write('grads: {}\n'.format(grads))
                for i, g in enumerate(grads):
                    if episode <= self.low_fidelity_iter:
                        if g<0 and not_max[i]: recommand.append(i)
                    else:
                        if not_max[i]: recommand.append(i)
                if self.params[5] < max(self.params[7], self.params[8], self.params[9]):
                    recommand = [5] # decode width should be larger than FU issue width  
                if (self.params[5] == 5 and self.params[10] < 8) or (self.params[5] >= 3 and self.params[10] < 4):
                    recommand = [10] # issue queue should fit decode width
                # with open(os.path.join(self.log_path, "details.txt"),"a") as f: f.write('recommand: {}\n'.format(recommand))
                if len(recommand) == 0:
                    break
                if episode <= self.warmup:
                    x=choice(recommand) # random pick one param to update.
                else:
                    action = [self.sample_from_truncated_normal_distribution(0, 1, y_pred[0][i].detach().numpy(), self.sigma, size=1)[0] for i in recommand]
                    x = recommand[torch.argmax(torch.tensor(action))]
                updates[x] = 1 # currently only allow increase
                norm = tdist.Normal(y_pred[0], self.sigma)
                log_prob_pool.append(sum(norm.log_prob(torch.tensor(updates))))
                
                not_max = self.arch_update(updates)
                metrics = self.evaluate()
                cpi = metrics[0]
                area = metrics[3]
                if area > self.area_limit: break
                
                cpi_improve = last_cpi-cpi # cpi smaller the better
                area_increase = max(area-last_area, 0.01)
                reward_step = min(float(cpi_improve/area_increase), 5) # limit the reward value in case area increase is to small 
                step_reward_pool.append(reward_step)
                
                last_cpi = float(cpi)
                last_area = float(area)
                last_params = self.current_params()
                
                self.record.store(step, round(float(reward_step),2), [round(float(x),2) for x in metrics], round(float(area),2), [round(x,2) for x in y_pred.data[0].tolist()], [int(x) for x in updates], self.current_params(), grads)
            # with open(os.path.join(self.log_path, "design.txt"),"a") as f:
            #     f.write("{}\n".format(last_params)) 
            self.recent_cpi.pop(0)
            self.recent_cpi.append(last_cpi)
            assert len(self.recent_cpi) == 5
            
            if episode <= self.low_fidelity_iter:
                # collect all lf cpi info and then rank them
                if last_cpi not in self.lf_cpi_pool: 
                    self.lf_cpi_pool.append(last_cpi)
                    sorted_pool = np.sort(self.lf_cpi_pool)
                # update best lf cpi
                if last_cpi < self.best_lf_cpi:
                    self.best_lf_cpi = last_cpi
                    self.lf_best_designs = []
                    self.lf_best_areas = []
                # collect best lf cpi info
                if (last_cpi == self.best_lf_cpi) and (last_params not in self.lf_best_designs):
                    self.lf_best_designs.append(last_params)
                    self.lf_best_areas.append(last_area)
                # calculate reward
                if episode > self.warmup:
                    top_epsilon = sorted_pool[0]+0.05
                    recent_avg = sum(self.recent_cpi)/len(self.recent_cpi)
                    if top_epsilon < self.baseline_lf_cpi:
                        self.baseline_lf_cpi = top_epsilon
                    # if recent_avg < self.baseline_lf_cpi:
                    #     self.baseline_lf_cpi = recent_avg
                    # if recent_avg < sorted_pool[1]:
                    #     self.baseline_lf_cpi = 0.5 * (sorted_pool[0] + sorted_pool[1])
                    # self.baseline_lf_cpi = sorted_pool[0]+0.05 # sorted_pool[0]+0.1 # 0.5 * (sorted_pool[0] + sorted_pool[1]) # 0.4
                    
                    reward_episode = float(100/last_cpi-100/self.baseline_lf_cpi) # + 0.1/(self.area_limit-last_area))
                    if reward_episode>0:
                        reward_episode = reward_episode * 10
                    # else: 
                    #     reward_episode = 0
                else:
                    reward_episode = float(100/last_cpi-100/self.baseline_lf_cpi)
            if episode == 200: print(self.lf_best_designs)
            if episode == self.low_fidelity_iter:
                # with open(os.path.join(self.log_path, "LF-Bests.txt"),"a") as f:
                #     for item in self.lf_best_designs:
                #         f.write("{}\n".format(item))
                self.baseline_hf_cpi = self.high_fidelity_evaluate(last_params)
                self.hf_design_pool.append(last_params)
                with open(os.path.join(self.log_path, "High-Fidelity.txt"),"a") as f:
                    f.write("episode{} design:{} hf-cpi:{} area:{}\n".format(episode, last_params, self.baseline_hf_cpi, last_area)) 
                descent_sorted_area_idx = np.argsort(self.lf_best_areas)[::-1]
                descent_sorted_design = [self.lf_best_designs[i] for i in descent_sorted_area_idx[:3]] # 3 / 5 
                hf_start_points = [self.lf_best_designs[i] for i in descent_sorted_area_idx[-5:]]
                print('descent_sorted_design', descent_sorted_design)
                for ds in descent_sorted_design:
                    high_fidelity_cpi = self.high_fidelity_evaluate(ds)
                    if ds not in self.hf_design_pool: 
                        self.hf_design_pool.append(ds)
                        with open(os.path.join(self.log_path, "High-Fidelity.txt"),"a") as f:
                            f.write("episode{} design:{} hf-cpi:{} area:{}\n".format(episode, ds, high_fidelity_cpi, last_area)) 
                
            if episode > self.low_fidelity_iter:
                if self.incremental: # incremental learning
                    if step <= 1: reward_episode = -10
                    else:
                        high_fidelity_cpi = self.high_fidelity_evaluate(last_params)
                        reward_episode = float(100/high_fidelity_cpi-100/self.baseline_hf_cpi)
                        if last_params not in self.hf_design_pool: 
                            print(last_params)
                            self.hf_design_pool.append(last_params)
                            with open(os.path.join(self.log_path, "High-Fidelity.txt"),"a") as f:
                                f.write("episode{} design:{} hf-cpi:{} area:{} reward:{} sigma:{}\n".format(episode, last_params, high_fidelity_cpi, last_area, reward_episode, self.sigma)) 
                        else: 
                            print('hf-tested, pass')
                        print('high fidelity result: {}, reward: {}'.format(high_fidelity_cpi, reward_episode))
                        
                        if high_fidelity_cpi < self.best_hf_cpi:
                            self.best_hf_cpi = high_fidelity_cpi
                else: # FNN guided random search
                    if last_cpi > self.best_lf_cpi:
                        reward_episode = float(100/last_cpi-100/self.baseline_lf_cpi)# - 79.247 # + 0.1/(self.area_limit-last_area))
                    else:
                        high_fidelity_cpi = self.high_fidelity_evaluate(last_params)
                        if last_params not in self.hf_design_pool: 
                            self.hf_design_pool.append(last_params)
                        else: 
                            print('hf-tested, pass')
                        reward_episode = float(100/high_fidelity_cpi-100/self.baseline_hf_cpi)
                        print('high fidelity result: {}, reward: {}'.format(high_fidelity_cpi, reward_episode))
                        with open(os.path.join(self.log_path, "High-Fidelity.txt"),"a") as f:
                            f.write("episode{} design:{} hf-cpi:{} area:{} reward:{} sigma:{}\n".format(episode, last_params, high_fidelity_cpi, last_area, reward_episode, self.sigma)) 
                        if high_fidelity_cpi < self.best_hf_cpi:
                            self.best_hf_cpi = high_fidelity_cpi
                
            for step_reward, surprise in zip(step_reward_pool, log_prob_pool):
                # importance = step_reward + reward_episode
                importance = reward_episode
                loss_pool.append(- surprise * importance)
            
            if len(loss_pool) == 0:
                for surprise in log_prob_pool:
                    importance = reward_episode
                    loss_pool.append(- surprise * importance)
            loss = sum(loss_pool)/len(loss_pool)
            loss.backward()
            self.optimizer.step()
            # if run_line_search:
            #     self.line_search(last_params)
            self.model.layer['consequent'].coeff.data.clamp_(0.00000001, 0.99999999)
            if episode > self.warmup: self.sigma = self.sigma * self.decay_rate
            if episode > self.low_fidelity_iter: self.sigma = self.hf_sigma # * self.hf_decay_rate**(episode-self.low_fidelity_iter)
            
            self.record.write(episode, last_cpi, last_area, reward_episode)
            
            # with open(os.path.join(self.log_path, "baseline.txt"),"a") as f:
            #     f.write("{}\n".format(self.baseline_lf_cpi)) 
            # with open(os.path.join(self.log_path, "sigma.txt"),"a") as f:
            #     f.write("{}\n".format(self.sigma)) 
            # if episode % 10 == 0:
            #     with open(os.path.join(self.log_path, "antecedents.txt"),"a") as f:
            #         f.write("episode{}:\n{}\n\n\n".format(episode, self.model.show_antecedents())) 
            #     with open(os.path.join(self.log_path, "consequents.txt"),"a") as f:    
            #         f.write("episode{}:\n{}\n\n\n".format(episode, self.model.show_consequents())) 
            if len(self.hf_design_pool) >= 10:
                break
            
            # ---------------- reset ---------------- #
            if self.incremental and episode >= self.low_fidelity_iter: # incremental learning
                self.params = torch.tensor(choice(descent_sorted_design), requires_grad=True, dtype=torch.float32)
            else:
                self.params = self.init_state.clone().detach() #torch.tensor([16, 2, 256, 2, 4, 2, 2, 1, 1, 1, 32], requires_grad=True, dtype=torch.float32)
                self.params.requires_grad = True
            self.system = self.set_system(self.params)
        with open(os.path.join(self.log_path, "High-Fidelity.txt"),"a") as f:
            f.write("high fidelity sample number: {}\n{}".format(len(self.hf_design_pool), self.hf_design_pool))
        print("high fidelity sample number: {}".format(len(self.hf_design_pool)))
        print("Best high fidelity sample: {}".format(self.best_hf_cpi))
        print("Training ends")
        torch.save(self.model, os.path.join(self.log_path, 'fnn.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.log_path, 'fnn-state.pt'))
        return 
    
    def set_system(self, params):
        system = {
            "L1LineSize": 64, #byte
            "L1sets": params[0],
            "L1ways": params[1],
            "L2LineSize": 64, #byte
            "L2sets": params[2],
            "L2ways": params[3],
            "L1latency": 4,
            "L2latency": 21,
            "DRAMlatency": 274,
            "issue_width": params[7]+params[8]+params[9],
            "mshr": params[4],
            "dispatch": params[5],
            "FUint": params[7],
            "FUfp": params[8],
            "FUmem": params[9],
            "FUcontrol": 1,
            "Cycle_int_op": 1,
            "Cycle_int_mul": 3,
            "Cycle_int_div": 18,
            "Cycle_fp_op": 3,
            "Cycle_fp_mul": 5,
            "Cycle_fp_div": 10,
            "BWCoreL1": 139586437120,
            "BWL1L2": 42949672960,
            "BWL2DRAM": 85899345920,
            "freq": 2e9,
            "frontpipe": 5,
            "ROB": params[6],
            "IQ": params[10],
        }
        return system
    
    # def incremental(self, ):
    
    def line_search(self, target_params):
        counter = 0
        print('\nStart line search, target:{}'.format(target_params))
        with open(os.path.join(self.log_path, "line_search.txt"),"a") as f:
            f.write("Start line search, target:{}\n".format(target_params)) 
        while True:
            self.params = self.init_state.clone().detach()
            self.params.requires_grad = True
            self.system = self.set_system(self.params)
            metrics = self.evaluate()
            cpi = metrics[0]
            area = metrics[3]
            last_cpi = float(cpi)
            last_area = float(area)
            not_max = [True for n in range(len(self.params))]
            for step in range(1, self.num_step+1):# self.args.max_iter+1
                print('\niter:', step)#, end="  "
                self.step = step
                updates = [0 for n in range(len(self.params))]
                with torch.no_grad():
                    y_pred = self.model(self.get_input(metrics[0], area))
                recommand = []
                for i in range(len(self.params)):
                    if not_max[i]: recommand.append(i)
                
                action = [self.sample_from_truncated_normal_distribution(0, 1, y_pred[0][i].detach().numpy(), 0.0001, size=1)[0] for i in recommand]
                x = recommand[torch.argmax(torch.tensor(action))]
                updates[x] = 1 # currently only allow increase
                not_max = self.arch_update(updates)
                metrics = self.evaluate()
                cpi = metrics[0]
                area = metrics[3]
                if area > self.area_limit: break
                
                last_cpi = float(cpi)
                last_area = float(area)
                last_params = self.current_params()
            print('params:{}'.format(last_params))
            with open(os.path.join(self.log_path, "line_search.txt"),"a") as f:
                f.write('params:{}\n'.format(last_params)) 
            counter += 1
            if last_cpi <= self.best_lf_cpi: #counter > 100:#last_params == target_params or 
                high_fidelity_cpi = self.high_fidelity_evaluate(last_params)
                if last_params not in self.hf_design_pool: 
                    self.hf_design_pool.append(last_params)
                with open(os.path.join(self.log_path, "High-Fidelity.txt"),"a") as f:
                    f.write("Line search -- design:{} hf-cpi:{} area:{}\n".format(last_params, high_fidelity_cpi, last_area)) 
                if high_fidelity_cpi <= self.best_hf_cpi:
                    self.best_hf_cpi = high_fidelity_cpi
                    break
            self.optimizer.step()
            if counter > 10:
                break
        print('Line search ends after {} iterations\n'.format(counter))
        with open(os.path.join(self.log_path, "line_search.txt"),"a") as f:
                f.write('params:{}\n'.format(last_params)) 
        return
    
    def non_zero(self, tensor):
        flag = False
        for n in tensor:
            if torch.is_nonzero(n):
                    flag = True
                    break
        return flag
    

if __name__ == '__main__':
    torch.manual_seed(1)
    # params = torch.tensor([64, 16, 4096, 16, 4, 16, 8, 5, 5, 5, 160], requires_grad=True, dtype=torch.float32)
    params = torch.tensor([16, 2, 256, 2, 1, 2, 1, 1, 1, 1, 32], requires_grad=True, dtype=torch.float32)
    a=Agent(params)
    a.optimize()