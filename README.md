This is the repo for our DAC24 accepted paper: 
Explainable Fuzzy Neural Network with Multi-Fidelity Reinforcement Learning for Micro-Architecture Design Space Exploration

### Pre-request:
* conda with python3.10 is needed. The downloader can be obtained from this url:
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

* McPAT is needed during the DSE, we have placed a pre-built McPAT under this directory: 
```sh
RISCV_Model_and_DSE/FNN_DSE/analytic/mcpat/mcpat-1.3.0
```
>[!NOTE] If our pre-built McPAT is incompatible with your local environment, please re-build McPAT in this directory. For detailed building instructions, please check the McPAT official repo: https://github.com/HewlettPackard/mcpat 

* VCS is needed during the DSE, please make sure vcs is callable in your commmand line.

### Usage:

First, set the envrionment variable: 
```sh
export chipyard_home="Absolute path to RISCV_Model_and_DSE/chipyard"
```

Then, activate the conda envrionment:
```sh
source $chipyard_home/env.sh
```
Now we can start the dse using FNN with MFRL:
```sh
python dse.py
```

Also, we can run the baselines for comparison.

To use BoomExplorer, run this scrpt:
```sh
python dse-dkg.py
```
To use Scalable Constraint BO, run this scrpt:
```sh
python dse-sc.py
```
To use BagGBRT, run this scrpt:
```sh
python dse-gbrt.py
```
To use ActBoost, run this scrpt:
```sh
python dse-ada.py
```
To use Random Forest, run this scrpt:
```sh
python dse-rf.py
```

>[!NOTE] We directly modify the sklearn and botorch files to implement the above algorithms. Thus, these baselines must be run in the conda environment provided in this repo. The official sklearn and botorch package can't support these algorithms.

A dataset is automatically built under **dataset/vcs**.
The DSE results are stored in the **log** directory, containing:

* The trained FNN named fnn.pt and its state-dict named fnn-state.pt. 
* details.txt shows the detaied DSE process. For example, the following log shows the starting point of this RL episode, then it shows how the design parameters and metrics change in each step of the markov process.
```sh
--- episode: 1 ---
Starting point:params: [16, 2, 128, 2, 2, 1, 32, 1, 1, 1, 2]
cpi: 4.738, miss1: 0.18, miss2: 0.45, area: 4.324
--iter:1
update: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
params: [16, 4, 128, 2, 2, 1, 32, 1, 1, 1, 2]
cpi: 4.654, miss1: 0.15, miss2: 1.78, area: 4.468
--iter:2
update: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
params: [16, 4, 128, 2, 4, 1, 32, 1, 1, 1, 2]
cpi: 4.508, miss1: 0.15, miss2: 1.78, area: 4.572
--iter:3
update: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
params: [16, 8, 128, 2, 2, 1, 32, 1, 1, 1, 2]
cpi: 4.576, miss1: 0.08, miss2: 2.50, area: 5.36
```
* final_cpi.txt records the low-fidelity cpi of the design obtained in each episode.
* High-Fidelity.txt records the high-fidelity cpi of the designs obtained in the high-fidelity phase.
* make_config.log is the generation log of chipyard.
 
To obtain the rules, check the jupyter notebook **get_rules.ipynb**.

The benchmarks cover the commonly used applications, which are 
* dijkstra, fft, stringsearch from mibench (https://github.com/embecosm/mibench)* and
* mm, qsort, vvadd from riscv-test (https://github.com/riscv-software-src/riscv-tests). 

### Customization
To use a custom benchmark, 

* First obtain the static profiling results using ibm-pisa (https://github.com/exabounds/ibm-pisa). 

* Then, put the profilings under **anlytic/benchmarks**, with a folder named after the benchmark. 

* Next, use the analytic/fitting.ipynb tool to fit a linear functon for the cache reuse distance, and add it into **analytic/Analytic_model_torch.py**. 

* Finally, include the profiling in **dse.py**.

