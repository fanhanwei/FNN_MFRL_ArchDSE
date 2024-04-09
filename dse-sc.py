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

import math
import warnings
from dataclasses import dataclass

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from hf.vcs import get_cpi_vcs
from analytic.Analytic_model_torch import McPAT, CPI
warnings.filterwarnings("ignore")

device = torch.device("cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}
design_space = [[4,5,6], [1,2,3,4], [7,8,9,10,11], [1,2,3,4], [1,2,3,4,5], 
               [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2], [1,2], [1,2,3,4,5]]
bounds = torch.tensor([[x[0] for x in design_space],
                    [x[-1] for x in design_space]])
dim = 11
lb, ub = bounds
batch_size = 1
max_cholesky_size = float("inf")  # Always use Cholesky
TEST = False

def parse_args():
    parser = argparse.ArgumentParser(description='Boom script')
    parser.add_argument('--n_sample', default='10', type=int, help='episode number for the whole training')
    parser.add_argument('--area_limit', default='8', type=float, help='limit of the area')
    parser.add_argument('--n_init', default='5', type=int, help='init points where action is randomly chosen')
    parser.add_argument('--seed', default='1', type=int, help='seed for torch and random') # 7 11
    parser.add_argument('--log_path', default=os.path.join(working_dir, 'logs/SC'), type=str, help='file path of log files')
    parser.add_argument('--h_param', default='1', type=float, help='h param')
    # parser.add_argument('--benchmark', default='mm-405060-456', type=str, help='RISCV toolchain benchmarks: dijkstra, dhrystone, median, mm, mt-matmul, mt-vvadd, multiply, pmp, qsort, rsort, spmv, towers, vvadd1000')
    return parser.parse_args()
args = parse_args()
args.log_path = fold_maker(args.log_path)
print('Logs stored in {}.'.format(args.log_path))
set_seed(args.seed)
with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
    f.write('seed:{}\n'.format(args.seed)) 
benchmarks = ['dijkstra', 'mm-405060-456', 'vvadd1000', 'qsort8192', 'fft', 'stringsearch']

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

def cpi(ds):
    
    params = param_regulator(ds)
    print(params)
    if TEST:
        inputs = torch.tensor(params, requires_grad=True, dtype=dtype)
        sys = set_system(inputs)
        ipc = 1/CPI(sys, alg)
    else:
        cpi_total = 0
        # with open(os.path.join(args.log_path, "hf-progress.txt"),"a") as f: f.write('\n{}\n'.format(params))
        for bm in benchmarks:
            cpi = get_cpi_vcs(params, bm, args.log_path)
            cpi_total += cpi
        cpi_avg = cpi_total / len(benchmarks)
        ipc = 1/cpi_avg
    print(' ipc: {:.3f}'.format(float(ipc)), end='\n')
    return ipc 

def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    unnorm = unnormalize(x, bounds)
    return cpi(unnorm)

def c1(x):  # Equivalent to enforcing that sum(x) <= 0
    inputs = torch.tensor(param_regulator(x), requires_grad=True, dtype=torch.float32)
    sys = set_system(inputs)
    # area = area_regression(sys)
    area = McPAT(sys, alg, 0.5, 0.5, 0.5)
    print(' area', float(area))
    return area-8

def eval_c1(x):
    return c1(unnormalize(x, bounds))

@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(1, **tkwargs) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))

def update_tr_length(state: ScboState):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state

def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()

def update_state(state, Y_next, C_next):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Pick the best point from the batch
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Update the length of the trust region according to the success and failure counters
    state = update_tr_length(state)
    return state

# Define example state
state = ScboState(dim=dim, batch_size=batch_size)
print(state)

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraint values
    batch_size,
    n_candidates,  # Number of candidates for Thompson sampling
    constraint_model,
    sobol: SobolEngine,
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Create the TR bounds
    best_ind = get_best_index_for_batch(Y=Y, C=C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    # Thompson Sampling w/ Constraints (SCBO)
    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points using Constrained Max Posterior Sampling
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

# Generate initial data

K_tilde = random_ted(args.n_init, design_space)
print('seed', args.seed)
readable_init = [param_regulator(x) for x in K_tilde]
print('init_designs', readable_init)
init_designs =torch.stack([normalize(torch.tensor(x, dtype=dtype), bounds) for x in K_tilde])
assert len(init_designs) == args.n_init
train_X = init_designs
train_Y = torch.tensor([eval_objective(x) for x in train_X], **tkwargs).unsqueeze(-1)
C1 = torch.tensor([eval_c1(x) for x in train_X], **tkwargs).unsqueeze(-1)
reward = [train_Y[i]*((C1[i]<=0).type_as(train_Y[i])) for i in range(args.n_init)]
with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
    for i in range(args.n_init):
        f.write("{} - {} reward:{}\n".format(i, readable_init[i], float(1/reward[i])))
best_init = float(1/max(reward))
print('best of init cpi: ', best_init)
with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
    f.write('best of init cpi:{}\n'.format(best_init))
# print(C1)

# Initialize TuRBO state
state = ScboState(dim, batch_size=batch_size)

# Note: We use 2000 candidates here to make the tutorial run faster.
# SCBO actually uses min(5000, max(2000, 200 * dim)) candidate points by default.
N_CANDIDATES = min(5000, max(2000, 200 * dim))
sobol = SobolEngine(dim, scramble=True, seed=1)

fit_arg={"max_attempts": 20}
def get_fitted_model(X, Y):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll, **fit_arg)

    return model
from botorch import settings
count = args.n_init
with settings.debug(True):
    while not state.restart_triggered and count < 10:  # Run until TuRBO converges
        count+=1
        # Fit GP models for objective and constraints
        model = get_fitted_model(train_X, train_Y)
        c1_model = get_fitted_model(train_X, C1)

        # Generate a batch of candidates
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            X_next = generate_batch(
                state=state,
                model=model,
                X=train_X,
                Y=train_Y,
                C=C1,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                constraint_model=c1_model,
                sobol=sobol,
            )
        # print('X_next', X_next)
        # Evaluate both the objective and constraints for the selected candidaates
        Y_next = torch.tensor([eval_objective(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        C1_next = torch.tensor([eval_c1(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        C_next = C1_next
        with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
            f.write("{} - {} reward:{}\n".format(count, param_regulator(unnormalize(X_next[0], bounds)), float(1/(Y_next*(C1_next<=0).type_as(Y_next)))))
        # Update TuRBO state
        state = update_state(state=state, Y_next=Y_next, C_next=C_next)

        # Append data. Note that we append all data, even points that violate
        # the constraints. This is so our constraint models can learn more
        # about the constraint functions and gain confidence in where violations occur.
        train_X = torch.cat((train_X, X_next), dim=0)
        train_Y = torch.cat((train_Y, Y_next), dim=0)
        C1 = torch.cat((C1, C1_next), dim=0)
        # print(train_X)
        # Print current status. Note that state.best_value is always the best
        # objective value found so far which meets the constraints, or in the case
        # that no points have been found yet which meet the constraints, it is the
        # objective value of the point with the minimum constraint violation.
        if (state.best_constraint_values <= 0).all():
            print(f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}\n")
        else:
            violation = state.best_constraint_values.clamp(min=0).sum()
            print(
                f"{len(train_X)}) No feasible point yet! Smallest total violation: "
                f"{violation:.2e}, TR length: {state.length:.2e}\n"
            )
# print('x_train: ', x_train)
# print('y_train: ', y_train)
# print('best cpi: ', 1/max(best_observed_ei))
# with open(os.path.join(args.log_path, "details.txt"),"a") as f: 
#     f.write('best cpi:{}'.format(1/max(best_observed_ei)))
   
# start = time.time()

# print('Time spent:', time.time()-start)
print('Logs stored in {}.'.format(args.log_path))
# print('Best epoch:{}, loss:{}, locations:{}'.format(best_epoch, best_info[0], best_info[1]))
# print('Best epoch: {}, info {}'.format(best_epoch, best_info), file=open(os.path.join(logdir, 'final.txt'), 'a'))