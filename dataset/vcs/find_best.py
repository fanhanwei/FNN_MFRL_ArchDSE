import os
os.chdir('/rshome/hanwei.fan/WorkSpace/FNN_DSE/analytic') 
import sys
sys.path.append(".") 
from Analytic_model_torch import CacheMissrate, McPAT
from get_alg_properties import get_alg_info

benchmark = 'vvadd1000' # 'dijkstra' 'dhrystone'

bench_path = '/rshome/hanwei.fan/WorkSpace/FNN_DSE/analytic/benchmarks'
if benchmark == 'dhrystone':
    alg = get_alg_info('./analytic/benchmarks/dhrystone/dry.out', './analytic/benchmarks/dhrystone/dry_br.out')
if benchmark == 'dijkstra':
    alg = get_alg_info(os.path.join(bench_path, 'dijkstra/dijkstra_full_profile.json'), os.path.join(bench_path, 'dijkstra/dijkstra_results.txt'))
if benchmark == 'vvadd1000':
    alg = get_alg_info(os.path.join(bench_path, 'vvadd1000/vvadd1000-pisa.out'), os.path.join(bench_path, 'vvadd1000/vvadd1000_results.txt'))

print(alg['totalInstructions'])

with open(os.path.join('/rshome/hanwei.fan/WorkSpace/FNN_DSE/analytic/mcpat/mcpat_database', "arch.txt"), 'r') as r:
    lines = r.readlines()
tested_designs = [[int(x) for x in l[1:-2].split(', ')] for l in lines]
with open(os.path.join('/rshome/hanwei.fan/WorkSpace/FNN_DSE/analytic/mcpat/mcpat_database', 'total_area.txt'), 'r') as r:
    tested_areas = r.readlines()
tested_areas = [float(x.strip()) for x in tested_areas]

with open('/rshome/hanwei.fan/WorkSpace/FNN_DSE/dataset/vcs/{}.txt'.format(benchmark), "r") as r:
    sim_data = r.readlines()
# high_fidelity_dict = {}
best_cpi = 10
best_design = None
best_area = 0
for item in sim_data:
    pair = item.strip().split('--')
    design = [int(x) for x in pair[0][1:-1].split(', ')]
    cpi = float(pair[1])
    # high_fidelity_dict[pair[0][1:-1]] = cpi
    arch = {
        "L1LineSize": 64, #byte
        "L1sets": design[0],
        "L1ways": design[1],
        "L2LineSize": 64, #byte
        "L2sets": design[2],
        "L2ways": design[3],
        "L1latency": 4,
        "L2latency": 40,
        "DRAMlatency": 274,
        "issue_width": design[7] + design[8] + design[9],
        "mshr": design[4],
        "dispatch": design[5],
        "FUint": design[8],
        "FUfp": design[9],
        "FUmem": design[7],
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
        "ROB": design[6],
        "IQ": design[10],
    }
    miss1 = 1.
    miss2 = 0.9
    area = McPAT(arch, alg, cpi, miss1, miss2)
    if cpi < best_cpi and area < 10:
        best_cpi = cpi
        best_area = area
        best_design = design
        print('best cpi: {}\n area: {}\n design: {}'.format(best_cpi, best_area, best_design))
    
print('Final -- best cpi: {}\n area: {}\n design: {}'.format(best_cpi, best_area, best_design))
    
if __name__ == '__main__':
    pass