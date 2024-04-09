from .Objects.BoomObjects import *
from .backends.boom import FixParams
import math
import time
import os
try:
    chipyard_home = os.environ['chipyard_home']
except:
    print("Please specify 'chipyard_home' as environment variable")
    exit()
project_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sim_path = os.path.join(project_home, 'dataset/boom_dataset')
sim_path = os.path.join(chipyard_home, 'sims/dse')
sim_output_path = os.path.join(project_home, 'dataset/vcs/raw')
target_dir = os.path.join(project_home, 'dataset/vcs')

def init_param():
    # create the system, set the name of the system
    # 创建系统， 设置待生成系统的名称
    system = System()
    system.name = 'BoomGshareDSE'
    system.chipyard_path = chipyard_home
    system.frequency = 10
    # Top level
    # 配置顶层
    system.tile = Tile()
    #Tile 参数可在此配置
    system.tile.num = 1               # tile 个数
    system.tile.User = True          # User mode
    system.tile.Supervisor = False    # Supervisor mode
    system.tile.XLen = 64             # 可选 32,64
    system.tile.debug = False          # 是否支持debug

    #Tile 组件
    system.tile.core = Core()
    '''Core 参数可在此配置'''
    system.tile.core.fetchWidth = 4                      # 每周期取指数  需是2的倍数
    system.tile.core.decodeWidth = 1                     # 每周期译指数  不可多于fetchWidth
    system.tile.core.numRobEntries = 32                  # reorder buffer 大小  需满足 numRobEntries % coreWidth == 0 
    ##################   issue Params   ##########################      // coreWidth is width of decode, width of integer rename, width of ROB, and commit width
    system.tile.core.mem_issueWidth =1                   #mem指令 可发射数
    system.tile.core.mem_numEntries =2                   #issue 队列大小
    system.tile.core.mem_dispatchWidth = system.tile.core.decodeWidth                #mem指令 可分发数
    system.tile.core.int_issueWidth =1                   #int指令 可发射数
    system.tile.core.int_numEntries =3                   #issue 队列大小
    system.tile.core.int_dispatchWidth = system.tile.core.decodeWidth                #int指令 可分发数
    system.tile.core.fp_issueWidth =1                    #fp指令 可发射数
    system.tile.core.fp_numEntries =2                    #issue 队列大小
    system.tile.core.fp_dispatchWidth = system.tile.core.decodeWidth                 #fp指令 可分发数
    ##############################################################
    system.tile.core.numLdqEntries = 8                   #load 队列大小   需满足 (numLdqEntries-1) > coreWidth
    system.tile.core.numStqEntries = 8                   #store 队列大小     需满足 (numStqEntries-1) > coreWidth
    system.tile.core.numIntPhysRegisters = 52            #整形物理寄存器文件大小   需满足  numIntPhysRegs >= (32 + coreWidth)
    system.tile.core.numFpPhysRegisters = 48             #浮点物理寄存器文件大小   需满足  numFpPhysRegs >= (32 + coreWidth)
    system.tile.core.maxBrCount = 8                      #可同时推断的分支数   需满足 maxBrCount >=2
    system.tile.core.numFetchBufferEntries = 8           #取值buffer大小
    system.tile.core.ftqEntries = 16                     #取值目标队列大小, 存储取得的PC和与采用的分支预测结果相关信号
    system.tile.core.enableBranchPrediction = False

    # 配置总线
    system.bus = Bus()
    system.bus.beatBytes = 8 #8 或 16
    system.bus.topo = 'crossbar' # default is 'crossbar', optional 'Ring' and 'doubleRing' and 'doubleRingTY'
    '''总线  参数可在此配置
    system.bus.frequency_sysbus = 100.0    # 可选bus frequency
    system.bus.frequency_membus = 100.0
    system.bus.frequency_ctrlbus = 100.0
    system.bus.frequency_peribus = 100.0
    system.bus.frequency_frontbus = 100.0
    '''

    system.tile.icache = L1ICache()     # 必须配备
    '''Icache 参数可在此配置'''
    system.tile.icache.nSets = 64           # set associativity    不超过64
    system.tile.icache.nWays = 4
    system.tile.icache.nTLBWays = 32        # TLB


    system.tile.dcache = L1DCache()     # 必须配备
    '''Dcache 参数可在此配置'''
    system.tile.dcache.nSets = 64               # set associativity
    system.tile.dcache.nWays = 4
    system.tile.dcache.nTLBWays = 32
    system.tile.dcache.nMSHR = 2                # miss 处理单元个数
    system.tile.dcache.numDCacheBanks = 1       # Bank个

    system.tile.fpu = FPU()
    system.tile.muldiv = MulDiv()

    system.tile.BPD = Gshare()
    ''' Gshare '''
    system.tile.BPD.name = 'Custom' + 'Gshare'
    system.tile.BPD.bpdMaxMetaLength = 200
    system.tile.BPD.globalHistoryLength = 16           #  全局历史长度
    system.tile.BPD.tage.tableInfo =  (64, system.tile.BPD.globalHistoryLength, 7)   #  PHT 的 nSet, 历史长度， Tag 长度
    #btb  分支目标buffer
    system.tile.BPD.btb.nSets = 64                     #  btb set数
    system.tile.BPD.btb.nWays = 2                       #  btb way数
    #bim  2位饱和计数器
    system.tile.BPD.bim.nSets = 256                    #  bim 计数器个数
    
    #L2 Cache
    system.l2cache = L2Cache()
    #L2 Cache 参数可在此配置
    system.l2cache.nBanks = 1               # Bank 个数
    system.l2cache.nWays = 2                # Way 个数
    system.l2cache.nSets = 256             # Set 个数
    system.l2cache.ntlbEntries = 128        # L2 Cache TLB
    system.l2cache.ntlbWays = 1

    return system

def param_update(system, param_vec):
    
    system.name = 'Boom'+str(param_vec).replace(', ', 'n').strip('[').strip(']')
    print(system.name)
    # Icache Params
    system.tile.icache.nSets = param_vec[0]           # set associativity    不超过64
    system.tile.icache.nWays = param_vec[1]

    # Dcache Params
    system.tile.dcache.nSets = param_vec[0]               # set associativity
    system.tile.dcache.nWays = param_vec[1]
    system.tile.dcache.nMSHR = param_vec[4]                # miss 处理单元个数
    
    # L2 Cache Params
    system.l2cache.nSets = param_vec[2]             # Set 个数
    system.l2cache.nWays = param_vec[3]                # Way 个数
    
    system.tile.core.decodeWidth = param_vec[5]                     # 每周期译指数  不可多于fetchWidth
    if system.tile.core.decodeWidth == 3 and param_vec[6] % 3 != 0:
        system.tile.core.numRobEntries = param_vec[5] * (param_vec[6] // param_vec[5] +1)
    elif system.tile.core.decodeWidth == 5 and param_vec[6] % 5 != 0:
        system.tile.core.numRobEntries = param_vec[5] * (param_vec[6] // param_vec[5] +1)
    else:
        system.tile.core.numRobEntries = param_vec[6]             # reorder buffer 大小  需满足 numRobEntries % coreWidth == 0 
        
    # issue Params   
    system.tile.core.mem_issueWidth =param_vec[7]                   #mem指令 可发射数
    system.tile.core.int_issueWidth =param_vec[8]                   #int指令 可发射数
    system.tile.core.fp_issueWidth =param_vec[9]                    #fp指令 可发射数
    
    system.tile.core.mem_numEntries =param_vec[10]                   #issue 队列大小
    system.tile.core.int_numEntries =int(system.tile.core.mem_numEntries *1.5)                 #issue 队列大小
    system.tile.core.fp_numEntries =system.tile.core.mem_numEntries                     #issue 队列大小
    
    system.tile.core.int_dispatchWidth = system.tile.core.decodeWidth                #int指令 可分发数
    system.tile.core.mem_dispatchWidth = system.tile.core.decodeWidth                #mem指令 可分发数
    system.tile.core.fp_dispatchWidth = system.tile.core.decodeWidth                 #fp指令 可分发数
    ##############################################################
    
    if system.tile.core.decodeWidth == 1:
        system.tile.core.fetchWidth = 4
        system.tile.core.numFetchBufferEntries = 16 # McPat require fetch buffer >=16, or it will throw error "no valid data array organizations found"
        system.tile.core.numIntPhysRegisters = 52
        system.tile.core.numFpPhysRegisters = 64 # McPat require fetch fpPhyRF >=64, or it will throw warning "Cache size must >=64"
    elif system.tile.core.decodeWidth == 2:
        system.tile.core.fetchWidth = 4
        system.tile.core.numFetchBufferEntries = 16
        system.tile.core.numIntPhysRegisters = 80
        system.tile.core.numFpPhysRegisters = 64
    elif system.tile.core.decodeWidth == 3:
        system.tile.core.fetchWidth = 8
        system.tile.core.numFetchBufferEntries = 24
        system.tile.core.numIntPhysRegisters = 100
        system.tile.core.numFpPhysRegisters = 96
    elif system.tile.core.decodeWidth == 4:
        system.tile.core.fetchWidth = 8
        system.tile.core.numFetchBufferEntries = 32
        system.tile.core.numIntPhysRegisters = 128
        system.tile.core.numFpPhysRegisters = 128
    elif system.tile.core.decodeWidth == 5:
        system.tile.core.fetchWidth = 8
        system.tile.core.numFetchBufferEntries = 40
        system.tile.core.numIntPhysRegisters = 128
        system.tile.core.numFpPhysRegisters = 128
    else:
        raise NotImplementedError

    return system


def validity_check(system):
    #####  check & fixed params #############
    instbits = 16 #(Boom use Compresses inst)
    coreInstBytes = instbits/8
    # CacheDatabits = system.bus.beatBytes * 8
    CacheBlockBytes = 64
    system.tile.icache.fetchBytes = int(system.tile.core.fetchWidth * coreInstBytes)     # fetch长度
    icacheBanks = 1 if (system.tile.icache.fetchBytes <= 8) else 2
    
    flag = (system.tile.core.int_issueWidth <= system.tile.core.decodeWidth) \
            and (system.tile.core.mem_issueWidth <= system.tile.core.decodeWidth) \
            and (system.tile.core.fp_issueWidth <= system.tile.core.decodeWidth)
    
    assert math.log2(system.tile.core.fetchWidth).is_integer()
    assert system.tile.core.fetchWidth >= 4 #Logic gets kind of annoying with fetchWidth = 2
    assert system.tile.core.decodeWidth <= system.tile.core.fetchWidth
    assert system.tile.core.numIntPhysRegisters >= (32 + system.tile.core.decodeWidth)
    assert system.tile.core.numFpPhysRegisters >= (32 + system.tile.core.decodeWidth)
    assert system.tile.core.maxBrCount >= 2
    assert (system.tile.core.numRobEntries % system.tile.core.decodeWidth) == 0
    assert (system.tile.core.numLdqEntries-1) > system.tile.core.decodeWidth
    assert (system.tile.core.numStqEntries-1) > system.tile.core.decodeWidth
    assert system.tile.icache.nSets <= 64 
    assert system.tile.dcache.nMSHR >= 2
    assert system.l2cache.nWays > 1
    assert math.log2(system.tile.dcache.numDCacheBanks).is_integer()
    if system.tile.dcache.numDCacheBanks > 1:
        assert system.tile.dcache.numDCacheBanks >= system.tile.core.mem_issueWidth
    assert system.tile.core.mem_issueWidth <= 2

    assert system.tile.core.numFetchBufferEntries > system.tile.core.fetchWidth
    assert (system.tile.core.numFetchBufferEntries % system.tile.core.decodeWidth) == 0
    if 'Tage' in system.tile.BPD.name:
        for item in system.tile.BPD.tage.tableInfo:
            assert item[1] <= system.tile.BPD.globalHistoryLength
    assert system.tile.icache is not None
    assert system.tile.dcache is not None
    
    return flag
    
def evaluate(system, params, benchmark, training_log_path, skip_chipyard=False):
    ##################################################################################
    if True:#args.sim
        begin = time.time()
        if not skip_chipyard:
            os.chdir(os.path.join(system.chipyard_path, 'sims/vcs'))
            os.system('make CONFIG='+system.name + '> {} 2>&1'.format(os.path.join(training_log_path, 'make_config.log'))) # > path/make_config.log
        chipyard_time = time.time() - begin
        # print('chipyard time: {:.2f} s'.format(chipyard_time))
        sim_name = os.path.join(sim_path, ('simv-chipyard.harness-' + system.name))
        begin = time.time()
        log_path = os.path.join(sim_output_path, '{}/{}_{}.txt'.format(benchmark, system.name, benchmark))
        os.system(sim_name + ' {}'.format(os.path.join(system.chipyard_path, "sims/{}.riscv".format(benchmark))) + ' > {} 2>&1'.format(log_path)) #运行 .riscv 程序
        sim_time = time.time() - begin
        with open(log_path, 'r') as r:
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
        print(system.name, end=' ')
        if cycle is None or inst is None:
            print('error!', system.name)
        elif cycle == -1:
            print('re-run benchmark!')
        elif cycle == float('inf'):
            print('invalid config! pipeline hung')
        else:
            cpi = cycle/inst
            print('cpi:', cpi)
    return cpi, chipyard_time, sim_time

def get_cpi_vcs(design, benchmark, log_path):
    tag = ''
    for n in design[:-1]:
        tag += str(n)+', '
    tag += str(design[-1])
    
    sys = init_param()
    params= [design[0], design[1], design[2], design[3], 
        design[4], #mshr
        design[5], #dispatch / decode
        design[6], # ROB
        design[9], design[7], design[8], # mem / int / fp
        design[10] # IQ
        ]
    sys = param_update(sys, params)
    assert validity_check(sys)
    sim_name = 'simv-chipyard.harness-' + sys.name
    
    hf_data_paths=os.path.join(target_dir, '{}.txt'.format(benchmark))
    with open(hf_data_paths, "r") as r:
        sim_data = r.readlines()
    _dict = {}
    for item in sim_data:
        pair = item.strip().split('--')
        _cpi = float(pair[1])
        _dict[pair[0][1:-1]] = _cpi
    
    try:
        cpi = _dict[tag]
        print('{} Found in high fidelity database, cpi={}'.format(benchmark, cpi))
        # with open(os.path.join(log_path, "hf-progress.txt"),"a") as f: f.write('{} Found in high fidelity database, cpi={}\n'.format(benchmark, cpi)) 
    except:
        print('{} not found, tag: {}\nRuning VLSI Flow'.format(benchmark, tag))
        # with open(os.path.join(log_path, "hf-progress.txt"),"a") as f: f.write('{} not found, Runing VLSI Flow\n'.format(benchmark)) 

        if os.path.exists(os.path.join(sim_path, sim_name)):
            print("Design exist: ", sys.name, "   SKIP chipyard! ")
            cpi, chipyard_time, sim_time = evaluate(sys, params, benchmark, log_path, skip_chipyard=True)
            print('{} bench time: {:.2f} s'.format(benchmark, sim_time))
            
        else:
            config_exist = False
            with open(os.path.join(chipyard_home, 'generators/boom/src/main/scala/common/config-mixins.scala'), 'r') as r:
                config_mixins = r.readlines()
            for line in config_mixins:
                if sys.name in line:
                    config_exist = True
                    break
            if not config_exist:
                FixParams(sys)
            cpi, chipyard_time, sim_time = evaluate(sys, params, benchmark, log_path)
            print('chipyard time: {:.2f} s'.format(chipyard_time))
            print('{} bench time: {:.2f} s'.format(benchmark, sim_time))
        with open(hf_data_paths, 'a') as w:
            w.write("{}--{}\n".format(design, cpi))
            
    return cpi   

    
    
if __name__ == '__main__':
    print(sim_path)
    print(os.environ['chipyard_home'])
    
    
    