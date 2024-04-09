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
from util.common import fold_maker, set_seed
from analytic.get_alg_properties import get_alg_info
bench_path = os.path.join(working_dir, 'analytic/benchmarks')
alg = get_alg_info(os.path.join(bench_path, 'fft/fft-pisa.out'), os.path.join(bench_path, 'fft/fft_results.txt'))

def parse_args():
    parser = argparse.ArgumentParser(description='Boom script')
    parser.add_argument('--n_sample', default='10', type=int, help='episode number for the whole training')
    parser.add_argument('--area_limit', default='8', type=float, help='limit of the area')
    parser.add_argument('--n_init', default='5', type=int, help='init points where action is randomly chosen')
    parser.add_argument('--seed', default='2', type=int, help='seed for torch and random') # 7 11
    parser.add_argument('--log_path', default=os.path.join(working_dir, 'logs/DKG'), type=str, help='file path of log files')
    parser.add_argument('--method', default='dkg', type=str, help='choose method: discrete kernel or dkg')
    parser.add_argument('--h_param', default='1', type=float, help='h param')
    # parser.add_argument('--benchmark', default='mm-405060-456', type=str, help='RISCV toolchain benchmarks: dijkstra, dhrystone, median, mm, mt-matmul, mt-vvadd, multiply, pmp, qsort, rsort, spmv, towers, vvadd1000')
    return parser.parse_args()

def random_ted(size, design_space, verbose=False):
    from random import randint
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

def evaluate(params, benchmarks, log_path, area_limit):
    from hf.vcs import get_cpi_vcs
    from analytic.Analytic_model_torch import McPAT, CPI
    ds = set_system(torch.tensor(params, dtype=torch.float64))
    area = McPAT(ds, alg, 0.5, 0.5, 0.5)
    if area > area_limit: 
        return 0, -1
    else:
        # cpi_avg = CPI(ds, alg)
        # reward = 1/cpi_avg
        
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
   
def BO_dkg(x_init, y_init, n_sample, design_space, benchmarks, log_path, area_limit, valid_pool, h_param):
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.optim import optimize_acqf
    from botorch.acquisition import ExpectedImprovement
    from botorch.models.gp_regression import SingleTaskDKG
    
    x_temp = [[x[i]/max(design_space[i]) for i in range(len(x))] for x in x_init]
    x_train = torch.tensor(x_temp, dtype=torch.float32)
    y_train = torch.tensor(y_init, dtype=torch.float32)
    counter = len(y_train.nonzero())
    print('init hf samples:', counter)
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('h: {}\n'.format(h_param)) 
        f.write('init hf samples:{}\n'.format(counter)) 
    
    best_observed_ei = []
    best_observed_ei.append(y_train.max().item())
    print('best of init cpi: ', 1/max(best_observed_ei))
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('best of init cpi:{}\n'.format(1/max(best_observed_ei)))
    model = SingleTaskDKG(x_train, y_train.unsqueeze(-1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    training_iterations = 60 
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.001)
    for iteration in range(1, 201):
        print(iteration, end=' ')
        with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
            f.write('{} - '.format(iteration))
        #################### fit the DKG models ####################
        model.train()
        model.likelihood.train()

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            
        model.eval()
        model.likelihood.eval()
        #################### end of fitting ####################
    
        #################### bo inference the next sample ####################
        EI = ExpectedImprovement(model=model, best_f=max(best_observed_ei)+h_param, maximize = True)#0.3
        candidates, _ = optimize_acqf(
            acq_function=EI,
            bounds= torch.tensor([[x[0]/x[-1] for x in design_space], [1 for x in design_space]], dtype=torch.float32),
            q=1,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        new_x = candidates.detach()
        params = new_x.squeeze()
        params = param_regulator([int(params[i]*max(design_space[i])) for i in range(len(params))])
        print('params:', params, end=' ')
        with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
            f.write('params:{} '.format(params))
        reward, cpi = evaluate(params, benchmarks, log_path, area_limit)
        print('reward:', round(reward, 4), 'cpi:', round(cpi, 4))
        with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
            f.write('reward:{}  '.format(round(reward, 4)))
            f.write('cpi:{}\n'.format(round(cpi, 4))) if cpi>0 else f.write('\n')
        new_y = torch.tensor([reward], dtype=torch.float32) # add output dimension
        #################### end of inferencing ####################
        
        #################### update GP training set ####################
        x_train = torch.cat([x_train, new_x])
        y_train = torch.cat([y_train, new_y])
        best_observed_ei.append(y_train.max().item())
        
        if new_y > 0 and params not in valid_pool: 
            counter += 1
            valid_pool.append(params)
        if counter >= n_sample: break
        states = model.state_dict()
        model = SingleTaskDKG(x_train, y_train.unsqueeze(-1))
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        model.load_state_dict(states)
        #################### end of updating ####################
        
    print('x_train: ', x_train)
    print('y_train: ', y_train)
    print('best cpi: ', 1/max(best_observed_ei))
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('best cpi:{}'.format(1/max(best_observed_ei)))
   
if __name__ == '__main__':
    args = parse_args()
    args.log_path = fold_maker(args.log_path)
    print('Logs stored in {}.'.format(args.log_path))
    set_seed(args.seed)
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('seed:{}\n'.format(args.seed)) 
    benchmarks = ['dijkstra', 'mm-405060-456', 'vvadd1000', 'qsort8192', 'fft', 'stringsearch']
    design_space = [[4,5,6], [1,2,3,4], [7,8,9,10,11], [1,2,3,4], [1,2,3,4,5], 
               [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2], [1,2], [1,2,3,4,5]]
    x = random_ted(args.n_init, design_space)
    # x=[[4, 2, 11, 1, 1, 5, 2, 3, 1, 1, 5],[5, 3, 9, 1, 1, 1, 2, 2, 2, 2, 3],[4, 2, 10, 3, 4, 5, 1, 2, 1, 1, 4]]
    init_designs = [param_regulator(ds) for ds in x]
    assert len(init_designs) == args.n_init
    print(init_designs)
    y=[]
    valid_pool = []
    for params in init_designs:
        start = time.time()
        reward, cpi = evaluate(params, benchmarks, args.log_path, args.area_limit)
        y.append(reward)
        if reward > 0: valid_pool.append(params)
        print('params:', params)
        print('reward:', reward)
        with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
            f.write('init param:{} reward:{} cpi:{}\n'.format(params, round(reward,4), round(cpi,4))) 
        print('Time spent:', time.time()-start, '\n')
    start = time.time()
    BO_dkg(x, y, args.n_sample, design_space, benchmarks, args.log_path, args.area_limit, valid_pool, args.h_param)
    
    print('Time spent:', time.time()-start)
    print('Logs stored in {}.'.format(args.log_path))
    # print('Best epoch:{}, loss:{}, locations:{}'.format(best_epoch, best_info[0], best_info[1]))
    # print('Best epoch: {}, info {}'.format(best_epoch, best_info), file=open(os.path.join(logdir, 'final.txt'), 'a'))