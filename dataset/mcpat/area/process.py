import os 
import sys

with open ('/rshome/hanwei.fan/WorkSpace/FNN_DSE/dataset/mcpat/area/arch-list.txt', 'r') as r:
    lines = r.readlines()
n = 0
# for l in lines:
#     n += 1
#     if n ==10000: exit()
print([[int(x) for x in l[1:-2].split(', ')] for l in lines])
        
# with open ('/rshome/hanwei.fan/WorkSpace/FNN_DSE/dataset/mcpat/area/arch-list.txt', 'w') as w:
#     for l in lines:
#         # print(l[:-4] + ' 1, 1, 1,' + l[-4:])
#         w.write(l[:-4] + ' 1, 1, 1,' + l[-4:])