import os
import sys
working_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_dir) # change working directory to the location of this file
sys.path.append(".") # Adds higher directory to python modules path.
import argparse
import time
import torch
import itertools
import numpy as np
from random import randint, choice
from skopt import Optimizer, Space
from skopt.space import Integer
from skopt.learning import GradientBoostingQuantileRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor_std
from util.common import fold_maker, set_seed
from analytic.get_alg_properties import get_alg_info

bench_path = os.path.join(working_dir, 'analytic/benchmarks')
alg = get_alg_info(os.path.join(bench_path, 'fft/fft-pisa.out'), os.path.join(bench_path, 'fft/fft_results.txt'))

def parse_args():
    parser = argparse.ArgumentParser(description='Boom script')
    parser.add_argument('--n_sample', default='9', type=int, help='episode number for the whole training')
    parser.add_argument('--area_limit', default='8', type=float, help='limit of the area')
    parser.add_argument('--n_init', default='5', type=int, help='init points where action is randomly chosen')
    parser.add_argument('--seed', default='1', type=int, help='seed for torch and random')
    parser.add_argument('--log_path', default=os.path.join(working_dir, 'logs/Ada'), type=str, help='file path of log files')
    # parser.add_argument('--benchmark', default='mm-405060-456', type=str, help='RISCV toolchain benchmarks: dijkstra, dhrystone, median, mm, mt-matmul, mt-vvadd, multiply, pmp, qsort, rsort, spmv, towers, vvadd1000')
    return parser.parse_args()

def evaluate(params, benchmarks, log_path, area_limit):
    from hf.vcs import get_cpi_vcs
    from analytic.Analytic_model_torch import McPAT, CPI
    ds = set_system(torch.tensor(params, dtype=torch.float64))
    area = McPAT(ds, alg, 0.5, 0.5, 0.5)
    if area > area_limit: 
        return 0, -1
    else:
        # cpi_avg = CPI(ds, alg)
        
        cpi_total = 0
        # with open(os.path.join(log_path, "hf-progress.txt"),"a") as f: f.write('\n{}\n'.format(params))
        for bm in benchmarks:
            cpi = get_cpi_vcs(params, bm, log_path)
            cpi_total += cpi
        cpi_avg = cpi_total / len(benchmarks)
        reward = 1/cpi_avg
        return float(reward), float(cpi_avg)

def param_regulator(params):
    params = [int(x) for x in params]
    # print('params', params, end='')
    design = []
    design.append(2**params[0])
    design.append(2**params[1])
    design.append(2**params[2])
    design.append(2**params[3])
    design.append(2*params[4])
    design.append(params[5])
    design.append(32*params[6])
    design.append(params[7])
    design.append(params[8])
    design.append(params[9])
    design.append(min(2**params[10], 24))
    
    max_FU = max(design[7], design[8], design[9])
    design[5] = max(design[5], max_FU) # decode width should be larger than FU issue width
    if design[5] == 5:
        design[10] = max(design[10], 8)
    elif design[5] >= 3:
        design[10] = max(design[10], 4)
    return design

def set_system(params):
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

def random_ted(size, design_space, verbose=False):

    from sklearn.gaussian_process.kernels import RBF
    K = list(itertools.product(*design_space))
    m = size # init training set size
    Nrted = 59 # according to original paper
    u = 0.1 # according to original paper
    length_scale = 0.1 # according to original paper

    f = RBF(length_scale=length_scale)

    def F_kk(K):
        dis_list = []
        for k_i in K:
            for k_j in K:
                dis_list.append(f(np.atleast_2d(k_i), np.atleast_2d(k_j)))
        return np.array(dis_list).reshape(len(K), len(K))

    K_tilde = []
    for i in range(m):
        M = [K[randint(0,len(K)-1)] for _ in range(Nrted)]
        M = M + K_tilde
        F = F_kk(M)
        if verbose: print(F)
        denoms=[F[-i][-i] + u for i in range(len(K_tilde))]
        for i in range(len(denoms)):
            for j in range(len(M)):
                for k in range(len(M)):
                    F[j][k] -= (F[j][i] * F[k][i]) / denoms[i]
        if verbose: print('----------------------------\n', F)
        assert len(M) == F.shape[0]
        k_i = M[np.argmax([np.linalg.norm(F[i])**2 / (F[i][i] + u) for i in range(len(M))])] # find i that maximaize norm-2(column i of F)
        K_tilde.append(k_i)
    if verbose: print(K_tilde)
    return K_tilde


if __name__ == '__main__':
    args = parse_args()
    args.log_path = fold_maker(args.log_path)
    print('Logs stored in {}.'.format(args.log_path))
    set_seed(args.seed)
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('seed:{}\n'.format(args.seed)) 
    design_space = [[4,5,6], [1,2,3,4], [7,8,9,10,11], [1,2,3,4], [1,2,3,4,5], 
               [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2], [1,2], [1,2,3,4,5]]
    benchmarks = ['dijkstra', 'mm-405060-456', 'vvadd1000', 'qsort8192', 'fft', 'stringsearch']
    x = random_ted(args.n_init, design_space)
    init_designs = [param_regulator(ds) for ds in x]
    assert len(init_designs) == args.n_init
    print(init_designs)
    y=[]
    valid_pool = []
    cpi_pool = []
    counter = 0
    for params in init_designs:
        start = time.time()
        reward, cpi = evaluate(params, benchmarks, args.log_path, args.area_limit)
        y.append(reward)
        if reward > 0: 
            counter += 1
            valid_pool.append(params)
            cpi_pool.append(cpi)
        print('params:', params)
        print('reward:', reward)
        with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
            f.write('init param:{} reward:{} cpi:{}\n'.format(params, round(reward,4), round(cpi,4))) 
        print('Time spent:', time.time()-start, '\n')
    nn1 = MLPRegressor(hidden_layer_sizes=(8,8), 
                        activation='logistic',
                        learning_rate='adaptive',
                        solver='sgd',
                        learning_rate_init=0.001,
                        momentum=0.5, 
                        random_state=args.seed,
                        max_iter=10000)
    
    nn2 = MLPRegressor(hidden_layer_sizes=(6,6),
                        activation='logistic',
                        learning_rate='adaptive',
                        solver='sgd',
                        learning_rate_init=0.001,
                        momentum=0.5,
                        random_state=args.seed,
                        max_iter=10000)

    ada1 = AdaBoostRegressor(
            estimator=nn1,
            learning_rate=0.001,
            n_estimators=20,
            random_state=args.seed,
        )
    
    ada2 = AdaBoostRegressor(
            estimator=nn2,
            learning_rate=0.001,
            n_estimators=20,
            random_state=args.seed,
        )
    ada1.fit(x, y)
    ada2.fit(x, y)
    all_design = list(itertools.product(*design_space))
    for iter in range(200):
        print(iter, '-', end='')
        U = [all_design[randint(0,len(all_design)-1)] for _ in range(3000)]
        cv_list = []
        for sample in U:
            pred1 = ada1._get_all_predict([sample])
            pred2 = ada2._get_all_predict([sample])
            pred_all = np.concatenate((pred1, pred2), axis=1)
            mu = np.mean(pred_all)
            std = np.std(pred_all)
            assert std != 0
            cv = mu/std
            cv_list.append(cv)
        ascend_cv_idx = np.argsort(cv_list)
        top10_cv_idx = ascend_cv_idx[-10:]
        next_x = choice(top10_cv_idx)
        params = param_regulator(U[next_x])
        reward, cpi = evaluate(params, benchmarks, args.log_path, args.area_limit)
        x.append(params)
        y.append(reward)
        print(params, 'reward:', round(reward,4), '  cpi:', round(cpi,4))
        with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
            f.write('{} - '.format(iter))
            f.write('params:{} '.format(params))
            f.write('reward:{}  '.format(round(reward, 4)))
            f.write('cpi:{}\n'.format(round(cpi, 4))) if cpi>0 else f.write('\n')

        if reward > 0 and params not in valid_pool: 
            counter += 1
            valid_pool.append(params)
            cpi_pool.append(cpi)
        if counter >= args.n_sample: break
        ada1.fit(x, y)
        ada2.fit(x, y)
    U_ = [list(x) for x in all_design]
    pred1 = ada1.predict(U_)
    pred2 = ada2.predict(U_)
    pred = (pred1+pred2)
    rank = np.argsort(pred)[::-1]
    assert rank[0] == np.argmax(pred)
    c = 0
    from analytic.Analytic_model_torch import area_regression
    
    for i in rank:
        params = param_regulator(U_[i])
        if params[0]*params[1] * params[2]*params[3] < 4194304 and params[1]<16 and area_regression(set_system(torch.tensor(params, dtype=torch.float32)))< 8:
            c+=1
            print(c, params)
            reward, cpi = evaluate(params, benchmarks, args.log_path, args.area_limit)
            if reward>0: break
        
    cpi_pool.append(cpi)
    print('final -', params, 'reward:', round(reward,4), '  cpi:', round(cpi,4))
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('final - ')
        f.write('params:{} '.format(params))
        f.write('reward:{}  '.format(round(reward, 4)))
        f.write('cpi:{}\n'.format(round(cpi, 4))) if cpi>0 else f.write('\n')
    print(valid_pool)
    print('best cpi: {}'.format(min(cpi_pool)))
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('best cpi: {}'.format(min(cpi_pool)))