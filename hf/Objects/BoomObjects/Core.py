class Core:
    name = 'boom'
    fetchWidth = 4
    decodeWidth = 2
    numRobEntries = 64
    ##################   issue Params   ##########################
    mem_issueWidth =1 #amount of things that can be issued
    mem_numEntries =12 #size of issue queue
    mem_dispatchWidth = decodeWidth #amount of things that can be dispatched
    int_issueWidth =2 #amount of things that can be issued
    int_numEntries =20 #size of issue queue
    int_dispatchWidth = decodeWidth #amount of things that can be dispatched
    fp_issueWidth =1 #amount of things that can be issued
    fp_numEntries =16 #size of issue queue
    fp_dispatchWidth = decodeWidth #amount of things that can be dispatched
    ##############################################################
    numLdqEntries = 16 #size of load queue
    numStqEntries = 16 #size of store queue
    numIntPhysRegisters = 80 #size of the integer physical register file
    numFpPhysRegisters = 64 #size of the floating point physical register file
    maxBrCount = 12 #number of branches we can speculate simultaneously
    numFetchBufferEntries = 16 #number of instructions that stored between fetch&decode
    enableAgePriorityIssue = True #两种issue order
    enablePrefetching = False #在cache miss 时预取下一行数据
    enableFastLoadUse = True #推测式快速load
    useVM = True
    useAtomics = True
    useAtomicsOnlyForIO = False
    PMPs = 8
    ftqEntries = 32 #FetchTargetQueue, Queue to store the fetch PC and other relevant branch predictor signals that are inflight in the processor.
    intToFpLatency = 2
    imulLatency = 3
    numDCacheBanks = 1
    def __setattr__(self, attrname, value):
        self.__dict__[attrname] = value