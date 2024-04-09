import os
import sys
working_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_dir) # change working directory to the location of this file
sys.path.append(".") # Adds higher directory to python modules path.
try:
    from .mcpat.mcpat_interface import *
except: 
    from mcpat.mcpat_interface import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
mcpat_path = os.path.join(working_dir, 'mcpat-1.3.0/mcpat')
xml_path = os.path.join(working_dir, 'mcpat_input.xml')
mcpat_out_path = os.path.join(working_dir, 'mcpat.txt')
with open(os.path.join(os.path.join(working_dir, 'mcpat_database'), "arch.txt"), 'r') as r:
    lines = r.readlines()
tested_designs = [[int(x) for x in l[1:-2].split(', ')] for l in lines]
with open (os.path.join(os.path.join(working_dir, 'mcpat_database'), 'total_area.txt'), 'r') as r:
    tested_areas = r.readlines()
tested_areas = [float(x.strip()) for x in tested_areas]
tested_length = len(tested_designs)

CacheLevel_L1=1
CacheLevel_L2=2
CacheLevel_DRAM=3

def CacheHitrate(cacheLevel, archProperties, algProperties):
    D0dreuse128 = ((1,0)) # ???   bench info
    def D0dreuse64_L1(x):   #对查表结果的多项式拟合
        if algProperties['alg_name'] == 'dhrystone':
            return torch.tensor(1., dtype=torch.float32)#1.
        elif algProperties['alg_name'] == 'dijkstra':
            return 0.29682698 + 0.2267687*x -0.0404381*x**2 +0.00247407*x**3
        elif algProperties['alg_name'] == 'qsort8192':
            return 9.46273016e-01+1.11705026e-02*x-1.14047619e-03*x**2+5.09259238e-05*x**3         
        elif algProperties['alg_name'] == 'mm-405060-456':
                if x <= 7: 
                    return -1.06830000e+00 + 2.06203333e+00*x -1.07691667e+00*x**2 + 2.78991667e-01*x**3 -3.58333333e-02*x**4 +  1.82500000e-03*x**5
                elif x<=9:
                    return 0.4406+0.0615*x
                elif x<=10:
                    return 0.9689+0.0028*x
                else:
                    return 1
        elif algProperties['alg_name'] == 'vvadd':
            return 0.88115 +0.003125*x
        elif algProperties['alg_name'] == 'vvadd1000':
            return 0.00733256*x + 0.87026047
        elif algProperties['alg_name'] == 'fft':
            if x <= 6:
                return 0.275 + 0.1202*x
            if x <= 9:
                return 7.243e-01 + 9.682e-02*x -1.1375e-02*x**2 + 4.45e-04*x**3
            else:
                return torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'stringsearch':
            if x <= 7:
                return 9.41433333e-01 + 6.49999998e-04*x
            if x <= 9:
                return 0.7577 + 0.0265*x
            else:
                return torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'basicmath_small':
            return torch.tensor(1., dtype=torch.float32)
        else:
            raise NotImplementedError("The reuse distance curve is not found for this benchmark")
    
    def D0dreuse64_L2(x):   #对查表结果的多项式拟合
        if algProperties['alg_name'] == 'dhrystone':
            return torch.tensor(1., dtype=torch.float32)#1.
        elif algProperties['alg_name'] == 'qsort8192':
            return 1.0049-6e-03*x+5e-04*x**2 if x<=10 else torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'dijkstra':
            if x < 10:
                return -0.08041667 + 0.10695*x
            elif x <= 12:
                return 0.95585 + 0.00355*x
            else:
                return torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'mm-405060-456':
            if x<=9:
                return 0.4406+0.0615*x
            elif x<=10:
                return 0.9689+0.0028*x
            else:
                return torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'vvadd1000':
            return 0.0369 *x + 0.6178 if x<=9 else torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'vvadd':
            return 0.5374 +0.0375*x if x<=11 else torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'fft':
            return 0.7577 + 0.0265*x if x <= 9 else torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'stringsearch':
            return 7.243e-01 + 9.682e-02*x -1.1375e-02*x**2 + 4.45e-04*x**3 if x <= 9 else torch.tensor(1., dtype=torch.float32)
        elif algProperties['alg_name'] == 'basicmath_small':
            return torch.tensor(1., dtype=torch.float32)
        else:
            raise NotImplementedError("The reuse distance curve is not found for this benchmark")

    effectiveSizeL1 = archProperties["L1sets"] * archProperties["L1ways"]
    effectiveSizeL2 = archProperties["L2sets"] * archProperties["L2ways"]

    hitrateL1 = D0dreuse64_L1(torch.log2(effectiveSizeL1))
    hitrateL2 = D0dreuse64_L2(torch.log2(effectiveSizeL2))

    relHitrateL1 = hitrateL1
    if hitrateL1 != 1:
        relHitrateL2 = (hitrateL2 - hitrateL1) / (1 - hitrateL1)
    else:
        relHitrateL2 = torch.tensor(1., dtype=torch.float32)

    if cacheLevel == CacheLevel_L1: return relHitrateL1
    elif cacheLevel == CacheLevel_L2: return relHitrateL2
    else: return 0

def CacheMissrate(cacheLevel, archProperties, algProperties):
    return 1 - CacheHitrate(cacheLevel, archProperties, algProperties)

def FiMem(cacheLevel, archProperties, algProperties):
    # F0mem = 0.14428 # 来不及的话就直接动手填上，初步实验不用改  GetTotalVMixFraction[GetAlgKeyValueCheck[algProperties, "F0mem"]];
    F0mem = algProperties["F0mem"]
    
    if cacheLevel == CacheLevel_L1: 
        return torch.tensor(F0mem, dtype=torch.float32)
    elif cacheLevel > CacheLevel_L1 and cacheLevel <= CacheLevel_DRAM: 
        return FiMem(cacheLevel-1, archProperties, algProperties)*CacheMissrate(cacheLevel-1, archProperties, algProperties)
    else:
        print("error: unknown cache level") 
        return 0

def FiMemHit(cacheLevel, archProperties, algProperties):
    if cacheLevel>=CacheLevel_L1 and cacheLevel<CacheLevel_DRAM: 
        return FiMem(cacheLevel, archProperties, algProperties) - FiMem(cacheLevel+1, archProperties, algProperties)
    elif cacheLevel == CacheLevel_DRAM: 
        return FiMem(cacheLevel, archProperties, algProperties)
    else:
        print("error: unknown cache level") 
        return 0

def CacheLiLatency(cacheLevel, archProperties, algProperties):
    if cacheLevel == CacheLevel_L1: return archProperties["L1latency"]
    elif cacheLevel == CacheLevel_L2: return archProperties["L2latency"]
    elif cacheLevel == CacheLevel_DRAM: return archProperties["DRAMlatency"]

def LPMemMSHR(archProperties, algProperties=None):
    FeL2 = FiMemHit(CacheLevel_L2, archProperties, algProperties)
    PeL2 = CacheLiLatency(CacheLevel_L2, archProperties, algProperties)
    FeDRAM = FiMemHit(CacheLevel_DRAM, archProperties, algProperties)
    PeDRAM = CacheLiLatency(CacheLevel_DRAM, archProperties, algProperties)
    n0MSHR = archProperties["mshr"]
    mem_insts = FeL2 * PeL2 + FeDRAM * PeDRAM
    constraint = n0MSHR/mem_insts if mem_insts != 0 else torch.tensor(float("Inf"), dtype=torch.float32)
    return constraint

def EventListInt(archProperties, algProperties, includeAddr=False): # 定值 无可导项
    # FeIntMul = 0.0146049          ### PISA
    FeIntMul = algProperties["F0intmul"]
    PeIntMul = archProperties["Cycle_int_mul"]
    # FeIntDiv = 0.0284007          ### PISA
    FeIntDiv = algProperties["F0intdiv"]
    PeIntDiv = archProperties["Cycle_int_div"]
    # FeIntOther = ((0.126713 + 0.118083) if includeAddr else 0.118083)-(FeIntMul + FeIntDiv)          ### PISA
    FeIntOther = ((algProperties["F0addr"] + algProperties["F0intOnly"]) if includeAddr else algProperties["F0intOnly"])-(FeIntMul + FeIntDiv)
    PeIntOther = archProperties["Cycle_int_op"]
    # Todo check int op latency for Boom
    return torch.tensor(FeIntMul * PeIntMul + FeIntDiv * PeIntDiv + FeIntOther * PeIntOther, dtype=torch.float32)

def EventListMem(archProperties, algProperties, grad=False):
    FeL1 = FiMemHit(CacheLevel_L1, archProperties, algProperties)
    PeL1 = CacheLiLatency(CacheLevel_L1, archProperties, algProperties)
    FeL2 = FiMemHit(CacheLevel_L2, archProperties, algProperties)
    PeL2 = CacheLiLatency(CacheLevel_L2, archProperties, algProperties)
    FeDRAM = FiMemHit(CacheLevel_DRAM, archProperties, algProperties)
    PeDRAM = CacheLiLatency(CacheLevel_DRAM, archProperties, algProperties)
    # PE 皆为定值 latency
    return FeL1 * PeL1 + FeL2 * PeL2 + FeDRAM * PeDRAM

def EventListFP(archProperties, algProperties): # 定值 无可导项
    # FeFPMul = 0.0000213904  ### PISA
    FeFPMul = algProperties["F0fpmul"]
    PeFPMul = archProperties["Cycle_fp_mul"]
    # FeFPDiv = 9.58626*10**-8  ### PISA
    FeFPDiv = algProperties["F0fpdiv"]
    PeFPDiv = archProperties["Cycle_fp_div"]
    # FeFPOther = 0.000035751 - (FeFPMul + FeFPDiv)  ### PISA
    FeFPOther = algProperties["F0fp"] - (FeFPMul + FeFPDiv)
    PeFPOther = archProperties["Cycle_fp_op"]
    return torch.tensor(FeFPMul * PeFPMul + FeFPDiv * PeFPDiv + FeFPOther * PeFPOther, dtype=torch.float32)

def EventListControl(archProperties, algProperties): # 定值  无可导项
    # FeControl = 0.215023  ### PISA
    FeControl = algProperties["F0control"]
    PeControl = 1 # 假设 control 指令只用1个cycle
    
    return torch.tensor(FeControl * PeControl, dtype=torch.float32)

def LPILP(archProperties, algProperties=None, show_warning=False):
    n0IQ = archProperties["IQ"]# issue queue len
    # ILP0WindowSize = 54  ### PISA
    ILP0WindowSize = algProperties["ILP0WindowSize"]
    if (n0IQ != ILP0WindowSize) and show_warning: 
        print('Warning: ILP values of the workload are for a window size of " {} " entries while the architecture is configured with an instruction queue of size " {} " entries'.format(n0IQ, ILP0WindowSize))
    # ILP = 9.281
    ILP = algProperties["ILP0"]
    sum_events = EventListMem(archProperties, algProperties)+EventListInt(archProperties, algProperties, True)+EventListFP(archProperties, algProperties)
    # event 评估 latency
    return ILP/sum_events

def LPILPType(type, archProperties, algProperties=None):
    # ILP_dict = {"int":10.2745, "mem":2.0439, "control":6.5548, "fp":1.0186}
    ILP_dict = {"int":algProperties["ILP0int"], "mem":algProperties["ILP0mem"], "control":algProperties["ILP0control"], "fp":algProperties["ILP0fp"]}
    ILPtype = ILP_dict[type]

    eventList_dict = {
        "int":EventListInt(archProperties, algProperties), 
        "mem":EventListMem(archProperties, algProperties), 
        "control":EventListControl(archProperties, algProperties), 
        "fp":EventListFP(archProperties, algProperties)
    }
    eventList = eventList_dict[type]
    constraint = ILPtype/eventList if eventList != 0 else torch.tensor(float("Inf"), dtype=torch.float32)
    return constraint

def ConstraintFU(type, archProperties, algProperties=None):
    n0_dict = {"int":archProperties["FUint"], "fp":archProperties["FUfp"], "mem":archProperties["FUmem"], "control":archProperties["FUcontrol"]}
    FUs = n0_dict[type]
    # F0_dict = {"int":0.451834, "fp":0.0000357501}
    # Since ctrl op is treated as int op in Boom, we add F0control to F0int
    F0_dict = {"int":algProperties["F0int"]+algProperties["F0control"], "fp":algProperties["F0fp"], "mem":algProperties["F0mem"], "control":algProperties["F0control"]}
    # Fraction = F0_dict[type] + (0.126713 if type == "int" else 0)
    Fraction = F0_dict[type] + (algProperties["F0addr"] if type == "mem" else 0)
    constraint = FUs/Fraction if Fraction != 0 else torch.tensor(float("Inf"), dtype=torch.float32)
    if not isinstance(constraint, torch.Tensor): constraint = torch.tensor(constraint, dtype=torch.float32)
    return constraint

def LPMemFU(archProperties, algProperties=None):
    return ConstraintFU("mem", archProperties, algProperties)
def LPIntFU(archProperties, algProperties=None):
    return ConstraintFU("int", archProperties, algProperties)
def LPFPFU(archProperties, algProperties=None):
    return ConstraintFU("fp", archProperties, algProperties)
def LPControlFU(archProperties, algProperties=None):
    return ConstraintFU("control", archProperties, algProperties)

#(* Determine the memory stall penalty: the time that issuing stops until that it starts again. Account for the cycles during a stall for which still work can be performed *)
def MemoryStallPenalty(cacheLevel, cyclesUntilStall, ipc, archProperties, algProperties=None):
    F = FiMemHit(cacheLevel, archProperties, algProperties)
    if F==0:
        return 0
    P = CacheLiLatency(cacheLevel, archProperties, algProperties)
    n0ROB = archProperties["ROB"]
    # Fevents = F / torch.max(1, torch.min(F * ipc * P, F * n0ROB))
    F_temp = (F * ipc * P) if F * ipc * P < F * n0ROB else F * n0ROB #(* Fraction of overlapping events *) 
    Fevents = (F / F_temp) if F_temp > 1 else F
    # print('Cachelevel{}: '.format(cacheLevel), ('ipc included' if F * ipc * P < F * n0ROB else 'rob included') if F_temp > 1 else 'F included', ', =0: {}'.format(P <= cyclesUntilStall))
    # print("F:{}, P:{}".format(F, P), "F*ipc*P:{}".format(F * ipc * P), "F*nROB:{}".format(F * n0ROB))
    return (0 if P <= cyclesUntilStall else (Fevents * (P - cyclesUntilStall)))

def solve_IPC(archProperties, algProperties=None, debug=False):
    # The constraints
    C_issue = archProperties['issue_width']
    C_mshr = LPMemMSHR(archProperties, algProperties)
    # 涉及cache
    C_ILP = LPILP(archProperties, algProperties)

    C_ILP_int = LPILPType("int",archProperties, algProperties)
    C_ILP_mem = LPILPType("mem",archProperties, algProperties)
    C_ILP_ctrl = LPILPType("control",archProperties, algProperties)
    C_ILP_fp = LPILPType("fp",archProperties, algProperties)

    C_FU_mem = LPMemFU(archProperties, algProperties)
    C_FU_int = LPIntFU(archProperties, algProperties)
    C_FU_control = LPControlFU(archProperties, algProperties)
    C_FU_fp = LPFPFU(archProperties, algProperties)

    if debug: 
        print("Level 1 Constraints:")
        print('  C_issue, C_mshr, C_ILP: {},\n  C_ILP_types(int-mem-ctl-fp): {},\n  C_FU_types(int-mem-fp): {}\n'.format((C_issue,C_mshr,C_ILP),(C_ILP_int,C_ILP_mem,C_ILP_ctrl,C_ILP_fp),(C_FU_int,C_FU_mem,C_FU_fp)))
        # print('C_issue: {}, C_mshr: {}, C_ILP: {},\nC_ILP_types(int-mem-ctl-fp): {},\nC_FU_types(int-mem-ctl-fp): {}'.format(C_issue.dtype,C_mshr.dtype,C_ILP.dtype,(C_ILP_int.dtype,C_ILP_mem.dtype,C_ILP_ctrl.dtype,C_ILP_fp.dtype),(C_FU_int.dtype,C_FU_mem.dtype,C_FU_control.dtype,C_FU_fp.dtype)))
    
    C_list = torch.stack((C_issue, C_mshr, C_ILP, C_ILP_int, C_ILP_mem, C_ILP_ctrl, C_ILP_fp, C_FU_int, C_FU_mem, C_FU_fp)) # , C_FU_control is wiped out as ctrl ops are treated as int op
    # print('Constraint list: ', ', '.join(['%.2f'% x for x in C_list]))
    return torch.min(C_list), torch.argmin(C_list)

def CalculateIQROBStallCycles(ipc, archProperties, algProperties=None, debug=False):
    FeL2 = FiMemHit(CacheLevel_L2, archProperties, algProperties)
    PeL2 = CacheLiLatency(CacheLevel_L2, archProperties, algProperties)
    FeDRAM = FiMemHit(CacheLevel_DRAM, archProperties, algProperties)
    PeDRAM = CacheLiLatency(CacheLevel_DRAM, archProperties, algProperties)
    n0IQ = archProperties["IQ"]# sum of issue queue(int mem fp)
    n0ROB = archProperties["ROB"] # To be changed
    # ilp = 9.281 #GetAlgKeyValueCheck[algProperties, "ILP0"];
    ilp = algProperties["ILP0"] #GetAlgKeyValueCheck[algProperties, "ILP0"];
    # (* First, we determine which stall occurs first *)
    # (* Cycles until ROB stall depends on the size of the ROB and the IPC *)
    cyclesUntilROBStall = n0ROB / ipc
    #(* TEMP CALCULTE WITH NEW METHOD *  这个原文没解释，大概是估了一下通常ROB里有多少空的 entry)
    # cyclesUntilROBStall = (n0ROB - (0.5 * n0ROB * max(1, min(FeDRAM * ipc * PeDRAM, FeDRAM * n0ROB)))/ilp)/ipc 
    
    #(* Determining the IQ stall is more difficult, as this depends on the frequency and penalty of events as well *)
    #(* We first calculate the effective number of instructions which need to be issued before the IQ fills with blocked instructions *)
    #(* For now, we only consider L2 and DRAM events, as these are long enough to cause such as a stall *)

    # (* Calculate the number of parallel events for increasing combinations of penalties *)
    parallel_event1 = ipc * (FeL2 * PeL2 + FeDRAM * PeDRAM)
    parallel_event2 = ipc * (FeDRAM * PeDRAM)
    if parallel_event1>0:
        instructionsUntilIQFill1 = ilp*n0IQ/parallel_event1
    else: 
        instructionsUntilIQFill1 = np.inf
    if parallel_event2>0:
        instructionsUntilIQFill2 = ilp*n0IQ/parallel_event2
    else: 
        instructionsUntilIQFill2 = np.inf

    #(* In case of frequeny events which stall the IQ, we need to correct for the instructions which remain in the IQ and cause the IQ to fill faster *)
    
    drainedIQInstruction1 = 1/FeL2 if FeL2>0 else np.inf
    drainedIQInstruction2 = 1/FeDRAM if FeDRAM>0 else np.inf
    
    # cyclesUntilIQStall_list = []
    # for i in range(len(instructionsUntilIQFill)):
    #     cyclesUntilIQStall_list.append(min(instructionsUntilIQFill[i],drainedIQInstructions[i]) / ipc)
    
    if debug: 
        print('L2 IQ included: {}, DRAM IQ included: {}'.format(instructionsUntilIQFill1 < drainedIQInstruction1, instructionsUntilIQFill2 < drainedIQInstruction2))
        print('L2', instructionsUntilIQFill1, drainedIQInstruction1, PeL2)
        print('DRAM', instructionsUntilIQFill2, drainedIQInstruction2, PeDRAM)
    cyclesUntilIQStall_1 = instructionsUntilIQFill1/ipc if instructionsUntilIQFill1 < drainedIQInstruction1 else drainedIQInstruction1/ipc
    cyclesUntilIQStall_2 = instructionsUntilIQFill2/ipc if instructionsUntilIQFill2 < drainedIQInstruction2 else drainedIQInstruction2/ipc
    #(* Select the first one for which the cycles until stalls is larger than it's smallest penalty. Otherwise, the smallest penalty has no influence on IQ stalls and we need to eveluate the next *)
    if cyclesUntilIQStall_1<=PeL2:
        cyclesUntilIQStall = cyclesUntilIQStall_1
    elif cyclesUntilIQStall_2<=PeDRAM:
        cyclesUntilIQStall = cyclesUntilIQStall_2
    else:
        cyclesUntilIQStall = np.inf #only happens when benchmark too small and all data in Cache1 leading to 0 miss
        if debug:
            print('cyclesUntilIQStall is inf. Benchmark too small cauing 0 cache miss, or may have mistakes.')
    if debug: print('cyclesUntilROBStall: ', cyclesUntilROBStall, 'cyclesUntilIQStall: ', cyclesUntilIQStall)
    if cyclesUntilROBStall <= cyclesUntilIQStall:
        #(* Only ROB stalls *)
        if debug: print('Only Rob Stalls')
        penaltyIQ = 0
        penaltyROB = MemoryStallPenalty(CacheLevel_L2, cyclesUntilROBStall, ipc, archProperties, algProperties) + MemoryStallPenalty(CacheLevel_DRAM, cyclesUntilROBStall, ipc, archProperties, algProperties)
    else:
        #(* Only  stalls *)
        if debug: print('Only IQ Stalls')
        penaltyIQ = MemoryStallPenalty(CacheLevel_L2, cyclesUntilIQStall, ipc, archProperties, algProperties)
        cpi = 1 / ipc + penaltyIQ
        cyclesUntilROBStall = n0ROB * cpi
        #(* TEMP CALCULTE WITH NEW METHOD *)
        # cyclesUntilROBStall=(n0ROB - (0.5 * n0ROB * max(1, min(FeDRAM * ipc * PeDRAM, FeDRAM * n0ROB)))/ilp)*cpi
        penaltyROB = MemoryStallPenalty(CacheLevel_DRAM, cyclesUntilROBStall, ipc, archProperties, algProperties)
    return penaltyIQ, penaltyROB

# (* Core to L1 cache bandwidth constraint *)
def LPBandwidthCoreToL1(archProperties, algProperties):
    Fe = FiMem(CacheLevel_L1, archProperties, algProperties)
    width = archProperties["L1LineSize"]    # L1 data cache granularity (cache line size). (*64-bit for Boom*)
    maxBW = archProperties["BWCoreL1"]      # "B0Dmo" Core to L1/L2 cache bandwidth.
    f0 = archProperties["freq"]             # "f0" Core clock speed.
    
    #(* Bandwidth is in B/s. f0 [c/s] * ipc [i/c] * Fe [1/i] * width [B] *)         # ipc = max_bandwidth/(f0*Fe*width) 
    return maxBW / (f0 * Fe * width)

#(* L2 to L3 cache bandwidth constraint *)
def LPBandwidthL2ToL3(archProperties, algProperties):
	Fe = FiMem(CacheLevel_L2, archProperties, algProperties)
	width = archProperties["L2LineSize"]    # Shared L2 cache granularity (cache line size).
	maxBW = archProperties["BWL1L2"]      # "B2Dmo" L1 cache to L2 cache bandwidth.
	f0 = archProperties["freq"]             # "f0" Core clock speed.
	
	# (* Bandwidth is in B/s. f0 [c/s] * ipc [i/c] * Fe [1/i] * width [B] *)        # ipc = max_bandwidth/(f0*Fe*width) 
	return maxBW / (f0 * Fe * width)

# (* Shared L2 to DRAM bandwidth constraint *)
def LPBandwidthL2ToDRAM(archProperties, algProperties):
    Fe = FiMem(CacheLevel_DRAM, archProperties, algProperties)
    width = archProperties["L2LineSize"]    # Shared L2 cache granularity (cache line size).
    maxBW = archProperties["BWL2DRAM"]      # "B2Dmo" L3 cache to DRAM bandwidth.
    f0 = archProperties["freq"]             # "f0" Core clock speed.
    # (* Bandwidth is in B/s. f0 [c/s] * ipc [i/c] * Fe [1/i] * width [B] *)        # ipc = max_bandwidth/(f0*Fe*width) 
    return maxBW / (f0 * Fe * width)

def SolveIPCLevel2Constraints(maxipc, archProperties, algProperties=None, debug=False):
    n0dispatch = archProperties["dispatch"]
    Constraint_list = [
        maxipc, 
        n0dispatch,
        LPBandwidthCoreToL1(archProperties, algProperties),
        LPBandwidthL2ToL3(archProperties, algProperties),
        LPBandwidthL2ToDRAM(archProperties, algProperties)]
    filtered_list = []
    for item in Constraint_list:
        if ~item.isinf(): filtered_list.append(item)
    ipcsolved = torch.min(torch.stack(filtered_list))
    if debug:
        print('Level 2 Constraints: l1-bottleneck: {:.4f} dispatch: {}, mem-bandwidths: {:.4f}, {:.4f}, {:.4f}'.format(float(maxipc), int(n0dispatch), 
            LPBandwidthCoreToL1(archProperties, algProperties),
            LPBandwidthL2ToL3(archProperties, algProperties),
            LPBandwidthL2ToDRAM(archProperties, algProperties)))
    return ipcsolved

# (* Calculate branch mis penalty *)
def d0branch(ipc, archProperties, algProperties=None):
   n0frontpipe = archProperties["frontpipe"]    # Core front-end pipeline depth.
   n0IQ = 60#archProperties["IQ"] Don't consider IQ value here, because the architecture is different
   #F0control = 0.215023 # "F0control" Fraction of control opearations
   F0control = algProperties["F0control"]
   
   #(* Returned value is a latency in _cycles_ (not seconds) *)
   # F0bestMispredict = 0.0172 # "F0bestMispredict"
   F0bestMispredict = algProperties["bestMispredictionRate"]
   branchModelFactor = 1.71 # (* Haswell model fit *)
   #(* TODO: We only account for the static front-end pipeline refil here. Eyerman, 2006, states that the branch resolution time (which can be approximated as the window drain time) is actually the biggest contributor to the penalty. See his work for models (probably requires more analysis from the application characterization *)
   return F0control * (n0frontpipe + n0IQ/ipc) * F0bestMispredict * branchModelFactor #(* + WindowDrainTime[archProperties, algProperties] - WindowDrainTime[archProperties, algProperties] *) (* Total penalty is front-end refil + window drain time, but for window drain time there are useful instructions being completed *)

#(* Performance in CPI *)
def CPI(archProperties, algProperties=None, debug=False):
    if debug: 
        print('app name: ', algProperties['alg_name'])
    ipc, bottleneck = solve_IPC(archProperties, algProperties, debug)
    if debug: 
        print("solve IPC: ", ipc, 'IPC bottleneck:', bottleneck.data.numpy())
    d0IQ, d0ROB = CalculateIQROBStallCycles(ipc, archProperties, algProperties, debug)
    if debug: print("IQ/ROB Stall Cycles: ", d0IQ, d0ROB)

    ipc = SolveIPCLevel2Constraints(1/(1/ipc + d0IQ + d0ROB), archProperties, algProperties, debug)
    if debug: print("IPC Level 2: ", ipc)

    cpi_with_branch = 1/ipc + d0branch(ipc, archProperties, algProperties)
    if debug: print("1/cpi + branch: ", 1/ipc, d0branch(ipc, archProperties, algProperties))

    return cpi_with_branch

def current_params(system):
    return [int(system["L1sets"]), int(system["L1ways"]), int(system["L2sets"]), int(system["L2ways"]), 
            int(system["mshr"]), int(system["dispatch"]), int(system["ROB"]), 
            int(system["FUint"]), int(system["FUfp"]), int(system["FUmem"]), int(system["IQ"])]

def McPAT(archProperties, algProperties, cpi, l1MissRate, l2MissRate):
    if current_params(archProperties) in tested_designs:
        idx = tested_designs.index(current_params(archProperties))
        total_area = tested_areas[idx]
        print('Found in database, id = {}, area = {}'.format(idx, total_area))
    else:
        generate_xml(archProperties, algProperties, cpi, l1MissRate, l2MissRate, xml_path)
        print("running McPAT ...")
        total_area = -1
        mcpat_count = 0
        while total_area == -1:
            if mcpat_count > 3:
                raise ValueError('Mcpat cannot work propoerly!')
            if mcpat_count > 0: print("  McPAT retry {} time".format(mcpat_count))
            os.system("{} -infile {} -print_level 5 > {}".format(mcpat_path, xml_path, mcpat_out_path))
            areas = get_metrics_from_ouput(mcpat_out_path)
            total_area = areas['all']
            mcpat_count += 1
        
        with open(os.path.join(os.path.join(working_dir, 'mcpat_database'), "areas.txt"),"a") as f:
            f.write("{}\n".format(areas)) 
        with open(os.path.join(os.path.join(working_dir, 'mcpat_database'), "total_area.txt"),"a") as f:
            f.write("{}\n".format(total_area))
        with open(os.path.join(os.path.join(working_dir, 'mcpat_database'), "arch.txt"),"a") as f:
            f.write("{}\n".format(current_params(archProperties))) 
        tested_designs.append(current_params(archProperties))
        tested_areas.append(total_area)
    return total_area

def area_regression(archProperties):
    A = torch.stack((archProperties["L1sets"], archProperties["L1ways"], archProperties["L2sets"], archProperties["L2ways"], 
                    archProperties["issue_width"], archProperties["mshr"], archProperties["dispatch"], archProperties["ROB"], 
                    archProperties["FUint"], archProperties["FUfp"], archProperties["FUmem"], torch.Tensor([1])[0]))
    x = torch.Tensor([3.42522271e-03,  2.13071929e-01,  2.72723973e-03,  7.35940210e-01,
        2.59108712e-01,  1.00000000e-04,  3.49348148e-05,  1.54640025e-03,
        1.12582433e-01,  2.72003123e-01,  4.44466353e-01, -2.68389346e+00])
    return torch.dot(A,x)

class area_MLP():       
    def __init__(self):
        pthfile = 'xxx.pth'
        self.area_model = torch.load(pthfile, map_location=torch.device('cpu'))
        
    def area_input(self, system):
        return torch.stack((system["L1sets"], system["L1ways"], system["L2sets"], system["L2ways"], 
                    system["issue_width"], system["mshr"], system["dispatch"], system["ROB"], 
                    system["FUint"], system["FUfp"], system["FUmem"]))
        
    def __call__(self, arch):
        return self.area_model(torch.Tensor(self.area_input(arch)))

if __name__ == '__main__':
    from get_alg_properties import get_alg_info
    # alg = get_alg_info('./benchmarks/dhrystone/dry.out', './benchmarks/dhrystone/dry_br.out')
    # alg = get_alg_info('./benchmarks/dijkstra/dijkstra_full_profile.json', './benchmarks/dijkstra/dijkstra_results.txt')
    # alg = get_alg_info('./benchmarks/vvadd/vvadd_full_profile.json', './benchmarks/vvadd/vvadd_results.txt')
    # alg = get_alg_info('./benchmarks/vvadd1000/vvadd1000-pisa.out', './benchmarks/vvadd1000/vvadd1000_results.txt')
    # alg = get_alg_info('./benchmarks/basicmath_small/basicmath_small-pisa.out', './benchmarks/basicmath_small/basicmath_small_results.txt')
    # alg = get_alg_info('./benchmarks/mm-405060-456/mm-pisa.out', './benchmarks/mm-405060-456/mm_results.txt') #7.5
    # alg = get_alg_info('./benchmarks/qsort8192/qsort8192-pisa.out', './benchmarks/qsort8192/qsort8192_results.txt')
    alg = get_alg_info('./benchmarks/fft/fft-pisa.out', './benchmarks/fft/fft_results.txt')
    # alg = get_alg_info('./benchmarks/stringsearch/stringsearch-pisa.out', './benchmarks/stringsearch/stringsearch_results.txt')
    # print(alg['F0int'], alg['F0fp'], alg['F0mem'])
    params = torch.tensor([64, 4, 256, 4, 2, 3, 64, 2, 1, 2, 2], requires_grad=True, dtype=torch.float32) #mshr dispatch ROB int fp mem IQ

    arch = {
        "L1LineSize": 64, #byte
        "L1sets": params[0],
        "L1ways": params[1],
        "L2LineSize": 64, #byte
        "L2sets": params[2],
        "L2ways": params[3],
        "L1latency": 4,
        "L2latency": 21,
        "DRAMlatency": 274,
        "issue_width": params[7] + params[8] + params[9],
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
    
    import time
    a = time.time()
    cpi = CPI(arch, alg, True)
    # print('time:', time.time()-a)
    # cpi.backward()
    # ipc = solve_IPC(arch, alg, True)[0]
    # d0IQ, d0ROB = CalculateIQROBStallCycles(ipc, arch, alg, True)
    # print('IQ, ROB:', d0IQ, d0ROB)
    # d0IQ.backward()
    # d0ROB.backward()
    # (1/(1/ipc + d0IQ + d0ROB)).backward()
    # print(1/(1/ipc + d0IQ + d0ROB))
    # ipc = SolveIPCLevel2Constraints(1/(1/ipc + d0IQ + d0ROB), arch, alg, True)
    # cpi = 1/ipc
    # cpi_with_branch = 1/ipc + d0branch(ipc, archProperties, algProperties)
    cpi.backward()
    print('grads: ', params.grad)
    # print("cpi:", cpi)
    
    with torch.no_grad():
        miss1 = CacheMissrate(1, arch, alg)
        miss2 = CacheMissrate(2, arch, alg)
    # area = area_regression(arch)
    # # cpi.backward()
    # # print(params.grad)
    
    area = McPAT(arch, alg, cpi, miss1, miss2)
    # area2 = area_regression(arch)
    # print("\ncpi: {:.3f}, area:{:.3f}, miss1:{:.3f}, miss2:{:.3f}".format(cpi))
    
    print("\ncpi: {:.3f}, area:{:.3f}, miss1:{:.3f}, miss2:{:.3f}".format(cpi, area, miss1, miss2))
    # print('area2', float(area2))
    # area = area_regression(arch)
    # params.grad.zero_()
    # area.backward()
    # print(params.grad)
    
    # class MLP(nn.Module):
    #     def __init__(self, A):
    #         super(MLP, self).__init__()
    #         self.fc1 = nn.Linear(11, 16)
    #         self.fc2 = nn.Linear(16, 16)
    #         self.fc3 = nn.Linear(16, 16)
    #         self.fc4 = nn.Linear(16, 1)

    #     def forward(self, x):
    #         x = self.fc1(x)
    #         x = F.relu(x)
    #         x = self.fc2(x)
    #         x = F.relu(x)
    #         x = self.fc3(x)
    #         x = F.relu(x)
    #         x = self.fc4(x)
    #         return x
    # area_mlp = area_MLP()
    # area1 = area_mlp(arch)
    # # params.grad.zero_()
    # area1.backward()
    # print(params.grad)
    # print('cpi: {:.2f}, area: {:.2f}, miss1: {:.2f}, miss2: {:.2f}'.format(cpi, area1, miss1, miss2))
    