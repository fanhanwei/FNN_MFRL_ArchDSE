import os
import sys
working_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_dir) # change working directory to the location of this file
sys.path.append(".") # Adds higher directory to python modules path.
import argparse
import torch
from skopt import Optimizer, Space
from skopt.space import Integer
from skopt.learning import GradientBoostingQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, BaggingRegressor_std
from util.common import fold_maker, set_seed
from analytic.get_alg_properties import get_alg_info
bench_path = os.path.join(working_dir, 'analytic/benchmarks')
alg = get_alg_info(os.path.join(bench_path, 'fft/fft-pisa.out'), os.path.join(bench_path, 'fft/fft_results.txt'))

def parse_args():
    parser = argparse.ArgumentParser(description='Boom script')
    parser.add_argument('--n_sample', default='10', type=int, help='episode number for the whole training')
    parser.add_argument('--area_limit', default='8', type=float, help='limit of the area')
    parser.add_argument('--n_init', default='5', type=int, help='init points where action is randomly chosen')
    parser.add_argument('--seed', default='1', type=int, help='seed for torch and random')
    parser.add_argument('--log_path', default=os.path.join(working_dir, 'logs/GBRT'), type=str, help='file path of log files')
    parser.add_argument('--method', default='dkg', type=str, help='choose method: discrete kernel or dkg')
    # parser.add_argument('--benchmark', default='mm-405060-456', type=str, help='RISCV toolchain benchmarks: dijkstra, dhrystone, median, mm, mt-matmul, mt-vvadd, multiply, pmp, qsort, rsort, spmv, towers, vvadd1000')
    return parser.parse_args()

def evaluate(params, benchmarks, log_path, area_limit):
    from hf.vcs import get_cpi_vcs
    from analytic.Analytic_model_torch import McPAT, CPI
    ds = set_system(torch.tensor(params, dtype=torch.float64))
    area = McPAT(ds, alg, 0.5, 0.5, 0.5)
    if area > area_limit: 
        return -1.1111, 5
    else:
        # cpi_avg = CPI(ds, alg)
        # reward = cpi_avg
        cpi_total = 0
        # with open(os.path.join(log_path, "hf-progress.txt"),"a") as f: f.write('\n{}\n'.format(params))
        for bm in benchmarks:
            cpi = get_cpi_vcs(params, bm, log_path)
            cpi_total += cpi
        cpi_avg = cpi_total / len(benchmarks)
        reward = cpi_avg
        return float(reward), float(cpi_avg)

def param_regulator(params):
    params = [int(x) for x in params]
    # print('params', params, end='')
    design = []
    design.append(2**(params[0]+4))
    design.append(2**(params[1]+1))
    design.append(2**(params[2]+7))
    design.append(2**(params[3]+1))
    design.append(2*(params[4]+1))
    design.append(params[5]+1)
    design.append(32*(params[6]+1))
    design.append(params[7]+1)
    design.append(params[8]+1)
    design.append(params[9]+1)
    design.append(min(2**(params[10]+1), 24))
    
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

if __name__ == '__main__':
    args = parse_args()
    args.log_path = fold_maker(args.log_path)
    print('Logs stored in {}.'.format(args.log_path))
    set_seed(args.seed)
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('seed: {}\n'.format(args.seed))
    transform = 'identity'
    problem_space = [
        Integer(low=0, high=2, prior='uniform', transform=transform, name="L1set"),
        Integer(low=0, high=3, prior='uniform', transform=transform, name="L1way"),
        Integer(low=0, high=4, prior='uniform', transform=transform, name="L2set"),
        Integer(low=0, high=3, prior='uniform', transform=transform, name="L2way"),
        Integer(low=0, high=4, prior='uniform', transform=transform, name="MSHR"),
        Integer(low=0, high=4, prior='uniform', transform=transform, name="decode_width"),
        Integer(low=0, high=4, prior='uniform', transform=transform, name="rob"),
        Integer(low=0, high=4, prior='uniform', transform=transform, name="int"),
        Integer(low=0, high=1, prior='uniform', transform=transform, name="fp"),
        Integer(low=0, high=1, prior='uniform', transform=transform, name="mem"),
        Integer(low=0, high=4, prior='uniform', transform=transform, name="IQ"),
        ]

    benchmarks = ['dijkstra', 'mm-405060-456', 'vvadd1000', 'qsort8192', 'fft', 'stringsearch']

    HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6, 'loss':'quantile'}
    HBO_params_ada_cpi = {'n_estimators': 20}
    # base_estimator = BaggingRegressor(
    #     base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
    #     **HBO_params_ada_cpi)
    gbrt = GradientBoostingRegressor(**HBO_params_cpi)
    gbrt_estimator = GradientBoostingQuantileRegressor(base_estimator=gbrt)
    base_estimator = BaggingRegressor_std(estimator=gbrt_estimator, **HBO_params_ada_cpi)
    opt = Optimizer(
        dimensions=problem_space,
        base_estimator=base_estimator,
        n_initial_points=args.n_init,
        initial_point_generator="TED",#orthogonal
        acq_func="LCB", #minimization version of UCB
        acq_optimizer="sampling",  # "auto",
        random_state=args.seed,
        n_jobs=-1,
        model_queue_size=1,
        acq_func_kwargs=None,  # {"xi": 0.000001, "kappa": 0.001} #favor exploitaton
        acq_optimizer_kwargs={"n_points": 10},
    )
    valid_pool=[]
    cpi_pool=[]
    counter = 0
    for iter in range(200):
        next_x = opt.ask()
        params = param_regulator(next_x)
        print(iter, '-', params, end=' ')
        reward, cpi = evaluate(params, benchmarks, args.log_path, args.area_limit)
        print('reward:', round(reward,4), '  cpi:', round(cpi,4))
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
        res = opt.tell(next_x, cpi)
    # print(valid_pool)
    print('best cpi: {}'.format(min(cpi_pool)))
    with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
        f.write('best cpi: {}'.format(min(cpi_pool)))
    