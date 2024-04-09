import os
import time
file_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.dirname(file_dir)
bench = 'vvadd1000'
parent_dir = os.path.join(file_dir, '{}'.format(bench))
start = time.time()
error_list = []
pipehung = []
dataset = []
redo=[]
with open(os.path.join(target_dir, 'vvadd1000.txt'), 'w') as w:
    w.write("")
for file_name in os.listdir(parent_dir):
    file_path = os.path.join(parent_dir, file_name)
    if os.path.isdir(file_path):
        continue
    if 'Boom' in file_path and 'vvadd1000' in file_path:
        design = [int(x) for x in file_name[4: file_name.find('_')].split('n')]
        design_ = design[:7] + design[8:10] + design[7:8] + design[10:] # adjust to the order of DSE codes
        # print(design, end=' ')
        with open(file_path, 'r') as r:
            contents = r.readlines()
        cycle = None
        inst = None
        for line in contents:
            if 'mcycle' in line:
                cycle = int(line[line.find('=')+2:].strip())
                # print(cycle)
            if 'minstret' in line:
                inst = int(line[line.find('=')+2:].strip())
                # print(inst)
            if 'Pipeline has hung' in line:
                cycle = float('inf')
                inst = 1
            if 'Received SIGHUP' in line:
                cycle = -1
                inst = 1
            
        if cycle is None or inst is None:
            error_list.append(design)
            # error_list.append(file_name)
            # print('error!')
        elif cycle == -1:
            # print('re-run benchmark!')
            redo.append(file_name)
        elif cycle == float('inf'):
            # print('invalid!')
            pipehung.append(file_name)
        else:
            cpi = cycle/inst
            # print('cpi:', cpi)
            dataset.append(design)
            with open(os.path.join(target_dir, 'vvadd1000.txt'), 'a') as w:
                w.write("{}--{}\n".format(design_, cpi))
# print('pipehung list length: {}'.format(len(pipehung)))
# print('error list:', 'length: {}'.format(len(error_list)))
# print(error_list)
# print('redo list:', 'length: {}'.format(len(redo)))
# print(redo)
print('vvadd dataset size: {}'.format(len(dataset)))
# print('running time: {}'.format(time.time()-start))