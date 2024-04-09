with open('/rshome/hanwei.fan/WorkSpace/FNN_DSE/dataset/vcs/dijkstra.txt', "r") as r:
    sim_data = r.readlines()
high_fidelity_dict = {}
for item in sim_data:
    pair = item.strip().split('--')
    # design = [int(x) for x in pair[0][1:-1].split(', ')]
    cpi = float(pair[1])
    high_fidelity_dict[pair[0][1:-1]] = cpi
params = [64, 16, 256, 16, 2, 2, 160, 1, 2, 1, 24]
tag = ''
for n in params[:-1]:
    tag += str(n)+', '
tag += str(params[-1])
try:
    cpi = high_fidelity_dict[tag]
    print(cpi)
    print(tag)
except:
    print('no')