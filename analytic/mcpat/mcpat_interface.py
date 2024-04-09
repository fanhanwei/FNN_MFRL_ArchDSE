import xml.etree.ElementTree as ET
from numpy import extract
import os
working_dir = os.path.dirname(os.path.abspath(__file__))

def generate_xml(arch, alg, cpi, l1MissRate, l2MissRate, path=os.path.join(working_dir, 'mcpat_input.xml')):

    LSys = float(alg['totalInstructions'])
    F0mem = float(alg['F0mem'])
    F0int = float(alg['F0int'])
    F0addr = float(alg['F0addr'])
    F0fp = float(alg['F0fp'])
    F0control = float(alg['F0control'])
    F0load = float(alg['F0load'])
    F0store = float(alg['F0store'])
    F0regreads = float(alg['F0regreads'])
    F0regwrites = float(alg['F0regwrites'])
    FMispredicted = float(alg['bestMispredictionRate'] * 1.71)

    cpi = float(cpi)
    l1MissRate = float(l1MissRate)
    l2MissRate = float(l2MissRate)

    issue_width = int(arch['issue_width'])
    dispatch = int(arch['dispatch'])
    
    if dispatch == 1:
        fetch_width = 4
        fetch_buffer = 16 # McPat require fetch buffer >=16, or it will throw error "no valid data array organizations found"
        intPhyRF = 52
        fpPhyRF = 64 # McPat require fetch fpPhyRF >=64, or it will throw warning "Cache size must >=64"
    elif dispatch == 2:
        fetch_width = 4
        fetch_buffer = 16
        intPhyRF = 80
        fpPhyRF = 64
    elif dispatch == 3:
        fetch_width = 8
        fetch_buffer = 24
        intPhyRF = 100
        fpPhyRF = 96
    elif dispatch == 4:
        fetch_width = 8
        fetch_buffer = 32
        intPhyRF = 128
        fpPhyRF = 128
    elif dispatch == 5:
        fetch_width = 8
        fetch_buffer = 40
        intPhyRF = 128
        fpPhyRF = 128
    else:
        raise NotImplementedError
        
    FUint = int(arch['FUint'])
    FUmem = int(arch['FUmem'])
    FUfp = int(arch['FUfp'])
    FUcontrol = int(arch['FUcontrol'])
    ROB = int(arch['ROB'])
    
    IQmem = int(arch['IQ'])
    IQfp = IQmem
    IQint = IQmem * 1.5
    
    L1_sets = int(arch['L1sets'])
    L1_ways = int(arch['L1ways'])
    L1_cache_size = L1_sets * L1_ways * 64

    L2_sets = int(arch['L2sets'])
    L2_ways = int(arch['L2ways'])
    L2_cache_size = L2_sets * L2_ways * 64
    
    mshr = int(arch['mshr'])

    tree = ET.parse(os.path.join(working_dir, 'mcpat-out.xml'))
    root = tree.getroot()
    #print("generating .xml file ...")
    for element in root.iter('stat'):# Element.findall()
        if element.get('name') == 'total_cycles':
            element.set('value',str(cpi*LSys))
        if element.get('name') == 'idle_cycles':
            element.set('value',str(cpi*LSys*0.001))
        if element.get('name') == 'busy_cycles':
            element.set('value',str(cpi*LSys))

        if element.get('name') == 'total_instructions':
            element.set('value',str(LSys)) 
        if element.get('name') == 'int_instructions':
            element.set('value',str((F0int+F0addr)*LSys)) 
        if element.get('name') == 'fp_instructions':
            element.set('value',str(F0fp*LSys)) 
        if element.get('name') == 'branch_instructions':
            element.set('value',str(F0control*LSys)) 
        if element.get('name') == 'branch_mispredictions':
            element.set('value',str(FMispredicted * F0control))
        if element.get('name') == 'load_instructions':
            element.set('value',str(F0load*LSys)) 
        if element.get('name') == 'store_instructions':
            element.set('value',str(F0store*LSys)) 
        if element.get('name') == 'committed_instructions':
            element.set('value',str(LSys)) 
        if element.get('name') == 'committed_int_instructions':
            element.set('value',str((F0int+F0addr)*LSys)) 
        if element.get('name') == 'committed_fp_instructions':
            element.set('value',str(F0fp*LSys)) 
        if element.get('name') == 'pipeline_duty_cycle':
            element.set('value',str(1/(cpi*issue_width))) 

        if element.get('name') == 'int_regfile_reads':
            element.set('value',str(F0regreads*(F0int+F0addr)*LSys)) 
        if element.get('name') == 'float_regfile_reads':
            element.set('value',str(F0regreads*F0fp*LSys)) 
        if element.get('name') == 'int_regfile_writes':
            element.set('value',str(F0regwrites*(F0int+F0addr)*LSys)) 
        if element.get('name') == 'float_regfile_writes':
            element.set('value',str(F0regwrites*F0fp*LSys)) 

        if element.get('name') == 'ialu_accesses':
            element.set('value',str((F0int+F0addr)*LSys)) 
        if element.get('name') == 'fpu_accesses':
            element.set('value',str(F0fp*LSys)) 

        if element.get('name') == 'memory_accesses':
            element.set('value',str((F0load+F0store)*LSys*l1MissRate*l2MissRate)) 
        if element.get('name') == 'memory_reads':
            element.set('value',str(F0load*LSys*l1MissRate*l2MissRate)) 
        if element.get('name') == 'memory_writes':
            element.set('value',str(F0store*LSys*l1MissRate*l2MissRate))             

    for element in root.iter('param'):# Element.findall()
        if element.get('name') == 'fetch_width':
            element.set('value',str(fetch_width))
        if element.get('name') == 'decode_width':
            element.set('value',str(dispatch))
        if element.get('name') == 'issue_width':
            element.set('value',str(issue_width))
        if element.get('name') == 'peak_issue_width':
            element.set('value',str(issue_width))
        if element.get('name') == 'commit_width':
            element.set('value',str(dispatch))

        if element.get('name') == 'fp_issue_width':
            element.set('value',str(FUfp))
        if element.get('name') == 'prediction_width':
            element.set('value',str(FUcontrol))
        if element.get('name') == 'ALU_per_core':
            element.set('value',str(FUint+FUmem))
        if element.get('name') == 'MUL_per_core':
            element.set('value',str(FUint))
        if element.get('name') == 'FPU_per_core':
            element.set('value',str(FUfp))

        if element.get('name') == 'instruction_buffer_size':
            element.set('value',str(fetch_buffer))
        if element.get('name') == 'instruction_window_size':
            element.set('value',str(IQmem + IQint))
        if element.get('name') == 'fp_instruction_window_size':
            element.set('value',str(IQfp))
        if element.get('name') == 'ROB_size':
            element.set('value',str(ROB))
            
        if element.get('name') == 'phy_Regs_IRF_size':
            element.set('value',str(intPhyRF))
        if element.get('name') == 'phy_Regs_FRF_size':
            element.set('value',str(fpPhyRF))

        if element.get('name') == 'icache_config':
            element.set('value', "{},64,{},1,1,2,64,0".format(L1_cache_size, L1_ways)) 
        if element.get('name') == 'dcache_config':
            element.set('value', "{},64,{},1,1,2,64,0".format(L1_cache_size, L1_ways))
        if element.get('name') == 'buffer_sizes' and element.get('value')[2:] == '4,4,4':
            element.set('value', "{},4,4,4".format(mshr)) 
        if element.get('name') == 'L2_config':
            element.set('value', "{},64,{},8,2,20,16,1".format(L2_cache_size, L2_ways))
        
    for element in root.findall('component/component/component/component[@name="icache"]/stat'):# icache stats
        if element.get('name') == 'read_accesses':
            element.set('value',str(LSys)) 
        if element.get('name') == 'read_misses':
            element.set('value',str(l1MissRate*LSys)) 

    for element in root.findall('component/component/component/component[@name="dcache"]/stat'):# icache stats
        if element.get('name') == 'read_accesses':
            element.set('value',str(F0load*LSys)) 
        if element.get('name') == 'write_accesses':
            element.set('value',str(F0store*LSys)) 
        if element.get('name') == 'read_misses':
            element.set('value',str(F0load*LSys*l1MissRate)) 
        if element.get('name') == 'write_misses':
            element.set('value',str(F0store*LSys*l1MissRate)) 

    for element in root.findall('component/component/component/component[@name="BTB"]/stat'):# icache stats
        if element.get('name') == 'read_accesses':
            element.set('value',str(F0control*LSys)) 

    for element in root.findall('component/component/component[@name="L20"]/stat'):# icache stats
        if element.get('name') == 'read_accesses':
            element.set('value',str(F0load*LSys*l1MissRate)) 
        if element.get('name') == 'write_accesses':
            element.set('value',str(F0store*LSys*l1MissRate))
        if element.get('name') == 'read_misses':
            element.set('value',str(F0load*LSys*l1MissRate*l2MissRate)) 
        if element.get('name') == 'write_misses':
            element.set('value',str(F0store*LSys*l1MissRate*l2MissRate))

    tree.write(path)        # 写回原文件

def get_metrics_from_ouput(filepath):
    area_dict={
        'all': -1,
        'core': -1,
        'l2': -1,
        'icache': -1,
        'dcache': -1,
        'ROB': -1,
        'FUint': -1,
        'FUfp': -1
    }
    with open(filepath,'r') as f:
        sourcelines=f.readlines()
        extract = False
        key = None
        error_msg = None
        for line in sourcelines:
            if extract:
                if 'Area' in line:
                    area_dict[key] = float(line[line.find('=')+2:][:line[line.find('=')+2:].find(' ')])
                    extract = False
            else:
                if 'Processor' in line:
                    extract = True
                    key = 'all'
                if 'Total Cores' in line:
                    extract = True
                    key = 'core'
                if 'Total L2s' in line:
                    extract = True
                    key = 'l2'
                if 'Instruction Cache' in line:
                    extract = True
                    key = 'icache'
                if 'Data Cache' in line:
                    extract = True
                    key = 'dcache'
                if 'ROB' in line:
                    extract = True
                    key = 'ROB'
                if 'Integer ALUs' in line:
                    extract = True
                    key = 'FUint'
                if 'Floating Point Units' in line:
                    extract = True
                    key = 'FUfp'
                if 'Error' in line:
                    error_msg = line
    if area_dict['all']==-1:
        import warnings
        warnings.warn('McPAT come into error -- {}'.format(error_msg), category=None, stacklevel=1, source=None)#
        print('McPAT come into error -- {}'.format(error_msg))
    return area_dict
                    
                    