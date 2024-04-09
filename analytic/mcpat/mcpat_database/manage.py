import os
with open("arch.txt", 'r') as r:
    lines = r.readlines()
tested_designs = [[int(x) for x in l[1:-2].split(', ')] for l in lines]
with open ('total_area.txt', 'r') as r:
    total_area = r.readlines()
with open ('areas.txt', 'r') as r:
    areas = r.readlines()
# tested_areas = [float(x.strip()) for x in tested_areas]
# tested_length = len(tested_designs)
for ds, ta, a in zip(tested_designs, total_area, areas):
    if ds[5] != 5:
        # with open("arch-new.txt", 'a') as f:
        #     f.write("{}\n".format(ds))
        with open("total_area-new.txt", 'a') as f:
            f.write("{}".format(ta))
        with open("areas-new.txt", 'a') as f:
            f.write("{}".format(a))