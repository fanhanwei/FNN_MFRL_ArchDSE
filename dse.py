import os
import sys
working_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_dir) # change working directory to the location of this file
sys.path.append(".") # Adds higher directory to python modules path.
import argparse
import time
import os
import torch
from util.common import fold_maker, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Boom script')
    parser.add_argument('--num_episode', default='500', type=int, help='episode number for the whole training')
    parser.add_argument('--num_step', default='20', type=int, help='step number for rach episode')
    parser.add_argument('--lr', default='0.01', type=float, help='learning rate')
    parser.add_argument('--area_limit', default='8', type=float, help='limit of the area')
    parser.add_argument('--warmup', default='100', type=int, help='warmup episodes where action is randomly chosen')
    parser.add_argument('--lf_iter', default='200', type=int, help='warmup episodes where action is randomly chosen')
    parser.add_argument('--sigma', default='0.3', type=float, help='variance of the guassian distribution for PG')
    parser.add_argument('--decay_rate', default='0.97', type=float, help='decay rate of random action sampling')
    parser.add_argument('--hf_sigma', default='0.3', type=float, help='variance of the guassian distribution for high-fidelity PG')
    parser.add_argument('--hf_decay_rate', default='0.95', type=float, help='decay rate of random action sampling')
    parser.add_argument('--seed', default='1', type=int, help='seed for torch and random')
    parser.add_argument('--method', default='MBPGMF', type=str, help='seed for torch and random')
    parser.add_argument('--opt', default='rmsprop', type=str, help='optimizer for NN updating: adam, rmsprop')
    parser.add_argument('--home_dir', default=working_dir, type=str, help='file path of home dir')    
    parser.add_argument('--log_path', default=os.path.join(working_dir, 'logs'), type=str, help='file path of log files')    
    parser.add_argument('--benchmark', default='mm-405060-456', type=str, help='RISCV toolchain benchmarks: dijkstra, dhrystone, median, mm, mt-matmul, mt-vvadd, multiply, pmp, qsort, rsort, spmv, towers, vvadd1000')
    return parser.parse_args()

def init():
    # create the system, set the name of the system
    params = torch.tensor([16, 2, 128, 2, 2, 1, 32, 1, 1, 1, 2], requires_grad=True, dtype=torch.float32)
    return params

if __name__ == '__main__':
    args = parse_args()
    args.log_path = os.path.join(os.path.join(args.log_path, args.method), args.benchmark)
    args.log_path = fold_maker(args.log_path)
    print('Logs stored in {}.'.format(args.log_path))
    # os.system('vivado -source /home/hanwei-fan/s2c/DSE/util/check_vivado.tcl')
    set_seed(args.seed)
    if args.method == 'MBPGMF':
        from agents.MB_PG_FNN_mf_avg import Agent
        
    params = init()
    
    from analytic.get_alg_properties import get_alg_info
    bench_path = os.path.join(working_dir, 'analytic/benchmarks')
    if args.benchmark == 'dijkstra': 
        dijkstra = get_alg_info(os.path.join(bench_path, 'dijkstra/dijkstra_full_profile.json'), os.path.join(bench_path, 'dijkstra/dijkstra_results.txt'))
        algs = [dijkstra]
    if args.benchmark == 'mm-405060-456': 
        mm = get_alg_info(os.path.join(bench_path, 'mm-405060-456/mm-pisa.out'), os.path.join(bench_path, 'mm-405060-456/mm_results.txt'))
        algs = [mm]
    if args.benchmark == 'vvadd1000': 
        vvadd = get_alg_info(os.path.join(bench_path, 'vvadd1000/vvadd1000-pisa.out'), os.path.join(bench_path, 'vvadd1000/vvadd1000_results.txt'))
        algs = [vvadd]
    if args.benchmark == 'qsort8192': 
        qsort = get_alg_info(os.path.join(bench_path, 'qsort8192/qsort8192-pisa.out'), os.path.join(bench_path, 'qsort8192/qsort8192_results.txt'))
        algs = [qsort]
    if args.benchmark == 'fft': 
        fft = get_alg_info(os.path.join(bench_path, 'fft/fft-pisa.out'), os.path.join(bench_path, 'fft/fft_results.txt'))
        algs = [fft]
    if args.benchmark == 'stringsearch': 
        stringsearch = get_alg_info(os.path.join(bench_path, 'stringsearch/stringsearch-pisa.out'), os.path.join(bench_path, 'stringsearch/stringsearch_results.txt'))
        algs = [stringsearch]
    if args.benchmark == 'all': 
        dijkstra = get_alg_info(os.path.join(bench_path, 'dijkstra/dijkstra_full_profile.json'), os.path.join(bench_path, 'dijkstra/dijkstra_results.txt'))
        mm = get_alg_info(os.path.join(bench_path, 'mm-405060-456/mm-pisa.out'), os.path.join(bench_path, 'mm-405060-456/mm_results.txt'))
        vvadd = get_alg_info(os.path.join(bench_path, 'vvadd1000/vvadd1000-pisa.out'), os.path.join(bench_path, 'vvadd1000/vvadd1000_results.txt'))
        qsort = get_alg_info(os.path.join(bench_path, 'qsort8192/qsort8192-pisa.out'), os.path.join(bench_path, 'qsort8192/qsort8192_results.txt'))
        fft = get_alg_info(os.path.join(bench_path, 'fft/fft-pisa.out'), os.path.join(bench_path, 'fft/fft_results.txt'))
        stringsearch = get_alg_info(os.path.join(bench_path, 'stringsearch/stringsearch-pisa.out'), os.path.join(bench_path, 'stringsearch/stringsearch_results.txt'))
        algs = [dijkstra, mm, vvadd, qsort, fft, stringsearch]
    
    start = time.time()
    agent = Agent(params=params, algs=algs, args=args)
    
    agent.optimize()
    print('Time spent:', time.time()-start)
    print('Logs stored in {}.'.format(args.log_path))