import os

def FixParams(system):
    boompath = os.path.join(system.chipyard_path, 'generators/boom/src/main/scala/common/config-mixins.scala')
    with open (boompath, 'r') as r:
        lines = r.readlines()
    with open (boompath, 'w') as w:
        for l in lines:
            # if '123456customize_marking' in l:
            #     break
            # else:
            #     w.write(l)
            if system.name in l:
                exist_flag = True
            w.write(l)
    boomconfig = open(boompath, 'a')
    #################################                 Tile                ################################
    boomconfig.write('\n')
    # boomconfig.write('//123456customize_marking\n')
    boomconfig.write('class WithN{}(n: Int = 1, overrideIdOffset: Option[Int] = None) extends Config(\n'.format(system.name))
    if system.tile.BPD is not None:
        boomconfig.write('  new With{} ++ // Default to TAGE-L BPD\n'.format(system.tile.BPD.name))
    else:
        boomconfig.write('  new WithTAGELBPD ++ // Default to TAGE-L BPD\n')
    boomconfig.write('  new Config((site, here, up) => {\n')
    boomconfig.write('    case TilesLocated(InSubsystem) => {\n')
    boomconfig.write('      val prev = up(TilesLocated(InSubsystem), site)\n')
    boomconfig.write('      val idOffset = overrideIdOffset.getOrElse(prev.size)\n')
    boomconfig.write('      (0 until n).map { i =>\n')
    boomconfig.write('        BoomTileAttachParams(\n')
    boomconfig.write('          tileParams = BoomTileParams(\n')
    boomconfig.write('            core = BoomCoreParams(\n')
    #################################      Core     ###############################  
    boomconfig.write('              fetchWidth = {},\n'.format(system.tile.core.fetchWidth))
    boomconfig.write('              decodeWidth = {},\n'.format(system.tile.core.decodeWidth))
    boomconfig.write('              numRobEntries = {},\n'.format(system.tile.core.numRobEntries))
    boomconfig.write('              issueParams = Seq(\n')
    boomconfig.write('                IssueParams(issueWidth={}, numEntries={}, iqType=IQT_MEM.litValue, dispatchWidth={}),\n'.format(
        system.tile.core.mem_issueWidth, system.tile.core.mem_numEntries, system.tile.core.mem_dispatchWidth))
    boomconfig.write('                IssueParams(issueWidth={}, numEntries={}, iqType=IQT_INT.litValue, dispatchWidth={}),\n'.format(
        system.tile.core.int_issueWidth, system.tile.core.int_numEntries, system.tile.core.int_dispatchWidth))
    boomconfig.write('                IssueParams(issueWidth={}, numEntries={}, iqType=IQT_FP.litValue, dispatchWidth={})),\n'.format(
        system.tile.core.fp_issueWidth, system.tile.core.fp_numEntries, system.tile.core.fp_dispatchWidth))
    boomconfig.write('              numLdqEntries = {},\n'.format(system.tile.core.numLdqEntries))
    boomconfig.write('              numStqEntries = {},\n'.format(system.tile.core.numStqEntries))
    boomconfig.write('              numIntPhysRegisters = {},\n'.format(system.tile.core.numIntPhysRegisters))
    boomconfig.write('              numFpPhysRegisters = {},\n'.format(system.tile.core.numFpPhysRegisters))
    boomconfig.write('              maxBrCount = {},\n'.format(system.tile.core.maxBrCount))
    boomconfig.write('              numFetchBufferEntries = {},\n'.format(system.tile.core.numFetchBufferEntries))
    boomconfig.write('              intToFpLatency = {},\n'.format(system.tile.core.intToFpLatency))
    boomconfig.write('              imulLatency = {},\n'.format(system.tile.core.imulLatency))
    boomconfig.write('              numDCacheBanks = {},\n'.format(system.tile.dcache.numDCacheBanks))
    if system.tile.core.enableAgePriorityIssue == False:
        boomconfig.write('              enableAgePriorityIssue = false,\n')
    if system.tile.core.enableAgePriorityIssue == True:
        boomconfig.write('              enableAgePriorityIssue = true,\n')
    else:
        boomconfig.write('              enableAgePriorityIssue = false,\n')
    if system.tile.core.enablePrefetching == True:
        boomconfig.write('              enablePrefetching = true,\n')
    else:
        boomconfig.write('              enablePrefetching = false,\n')
    if system.tile.core.enableFastLoadUse == True:
        boomconfig.write('              enableFastLoadUse = true,\n')
    else:
        boomconfig.write('              enableFastLoadUse = false,\n')
    if system.tile.core.useAtomicsOnlyForIO == True:
        boomconfig.write('              useAtomicsOnlyForIO = true,\n')
    else:
        boomconfig.write('              useAtomicsOnlyForIO = false,\n')
    #################################      Tile     ###############################
    #Mode
    if not system.tile.User:
        boomconfig.write('              useUser = false,\n')
    if system.tile.Supervisor:
        boomconfig.write('              useSupervisor = true,\n')
    #Virtual Memory (VIPT / PIPT)
    if system.tile.core.useVM == True:
        boomconfig.write('              useVM = true,\n')
    boomconfig.write('              nPMPs = {},\n'.format(system.tile.core.PMPs))
    #Debug
    if system.tile.debug == True:
        boomconfig.write('              useDebug = true,\n')
    #################################      L2 Cache  (optional)    Part I    ################################
    if system.l2cache is not None:
        boomconfig.write('              nL2TLBEntries = {},\n'.format(system.l2cache.ntlbEntries))
        boomconfig.write('              nL2TLBWays = {},\n'.format(system.l2cache.ntlbWays))
    #Other components
    boomconfig.write('              ftq = FtqParameters(nEntries={}),\n'.format(system.tile.core.ftqEntries))
    #################################      FPU  (optional)     ################################
    if system.tile.fpu is not None:
        boomconfig.write('              fpu = Some(freechips.rocketchip.tile.FPUParams(sfmaLatency={}, dfmaLatency={}, divSqrt={})),\n'.format(
            system.tile.fpu.fmaLatency, system.tile.fpu.fmaLatency, 'true' if system.tile.fpu.divsqrt else 'false'))
    #################################      Mul/Div  (optional)     ################################
    if system.tile.muldiv is not None:
        boomconfig.write('              mulDiv = Some(freechips.rocketchip.rocket.MulDivParams(mulUnroll={}, divUnroll={}, mulEarlyOut={}, divEarlyOut={})),\n'.format(
            system.tile.muldiv.mul_unroll, system.tile.muldiv.div_unroll, 'true' if system.tile.muldiv.mul_earlyout else 'false', 'true' if system.tile.muldiv.div_earlyout else 'false'))
    boomconfig.write('            ),\n')
    #################################     L1 Data Cache  (must)     ###############################
    boomconfig.write('            dcache = Some(\n')
    boomconfig.write('              DCacheParams(rowBits = site(SystemBusKey).beatBits, nSets={}, nWays={}, nTLBWays={}, tagECC=Some("{}"), dataECC=Some("{}"), nMSHRs={}, replacementPolicy="{}")\n'.format(
        system.tile.dcache.nSets, system.tile.dcache.nWays, system.tile.dcache.nTLBWays, system.tile.dcache.eccCode, system.tile.dcache.eccCode, system.tile.dcache.nMSHR, system.tile.dcache.replacement))
    boomconfig.write('            ),\n')
    #################################     L1 Inst Cache  (must)     ###############################
    boomconfig.write('            icache = Some(\n')
    boomconfig.write('              ICacheParams(rowBits = site(SystemBusKey).beatBits, nSets={}, nWays={}, nTLBWays={}, tagECC=Some("{}"), dataECC=Some("{}"), prefetch={}, fetchBytes={})\n'.format(
        system.tile.icache.nSets, system.tile.icache.nWays, system.tile.icache.nTLBWays, system.tile.icache.eccCode, system.tile.icache.eccCode, \
        'true' if system.tile.icache.prefetch else 'false', system.tile.icache.fetchBytes))
    boomconfig.write('            ),\n')
    ###############################################################################################
    boomconfig.write('            hartId = i + idOffset\n')
    boomconfig.write('          ),\n')
    boomconfig.write('          crossingParams = RocketCrossingParams()\n')
    boomconfig.write('        )\n')
    boomconfig.write('      } ++ prev\n')
    boomconfig.write('    }\n')
    boomconfig.write('    case SystemBusKey => up(SystemBusKey, site).copy(beatBytes = {})\n'.format(system.bus.beatBytes))
    ##############################################################################
    #Instruction Length
    if system.tile.XLen == 64:
        boomconfig.write('    case XLen => 64\n')
    if system.tile.XLen == 32:
        boomconfig.write('    case XLen => 32\n')
    boomconfig.write('  })\n')
    boomconfig.write(')\n')
    #################################      Branch Predict  (optional)     ################################
    if False: #system.tile.BPD is not None:
        if 'Tage' in system.tile.BPD.name:
            boomconfig.write('\nclass With{} extends Config((site, here, up) => '.format(system.tile.BPD.name))
            boomconfig.write('{\n')
            boomconfig.write('  case TilesLocated(InSubsystem) => up(TilesLocated(InSubsystem), site) map {\n')
            boomconfig.write('    case tp: BoomTileAttachParams => tp.copy(tileParams = tp.tileParams.copy(core = tp.tileParams.core.copy(\n')
            boomconfig.write('      bpdMaxMetaLength = {},\n'.format(system.tile.BPD.bpdMaxMetaLength))
            boomconfig.write('      globalHistoryLength = {},\n'.format(system.tile.BPD.globalHistoryLength))
            boomconfig.write('      localHistoryLength = {},\n'.format(system.tile.BPD.localHistoryLength))
            boomconfig.write('      localHistoryNSets = {},\n'.format(system.tile.BPD.localHistoryNSets))
            boomconfig.write('      branchPredictor = ((resp_in: BranchPredictionBankResponse, p: Parameters) => {\n')
            boomconfig.write('        val loop = Module(new LoopBranchPredictorBank()(p))\n')
            boomconfig.write('        val tage = Module(new TageBranchPredictorBank(\n')
            boomconfig.write('          BoomTageParams(tableInfo=Seq{}, uBitPeriod={}))(p))\n'.format(system.tile.BPD.tage.tableInfo, system.tile.BPD.tage.uBitPeriod))
            boomconfig.write('        val btb = Module(new BTBBranchPredictorBank(BoomBTBParams(nSets={}, nWays={}, offsetSz={}, extendedNSets={}))(p))\n'.format(system.tile.BPD.btb.nSets, system.tile.BPD.btb.nWays, system.tile.BPD.btb.offsetSz, system.tile.BPD.btb.extendedNSets))
            boomconfig.write('        val bim = Module(new BIMBranchPredictorBank(BoomBIMParams(nSets={}))(p))\n'.format(system.tile.BPD.bim.nSets))
            boomconfig.write('        val ubtb = Module(new FAMicroBTBBranchPredictorBank(BoomFAMicroBTBParams(nWays={}, offsetSz={}))(p))\n'.format(system.tile.BPD.ubtb.nWays, system.tile.BPD.ubtb.offsetSz))
            boomconfig.write('        val preds = Seq(loop, tage, btb, ubtb, bim)\n')
            boomconfig.write('        preds.map(_.io := DontCare)\n')
            boomconfig.write('        ubtb.io.resp_in(0)  := resp_in\n')
            boomconfig.write('        bim.io.resp_in(0)   := ubtb.io.resp\n')
            boomconfig.write('        btb.io.resp_in(0)   := bim.io.resp\n')
            boomconfig.write('        tage.io.resp_in(0)  := btb.io.resp\n')
            boomconfig.write('        loop.io.resp_in(0)  := tage.io.resp\n')
            boomconfig.write('        (preds, loop.io.resp)\n')
            boomconfig.write('      })\n')
            boomconfig.write('    )))\n')
            boomconfig.write('    case other => other\n')
            boomconfig.write('  }\n')
            boomconfig.write('})\n')
        elif 'Gshare' in system.tile.BPD.name:
            boomconfig.write('\nclass With{} extends Config((site, here, up) => '.format(system.tile.BPD.name))
            boomconfig.write('{\n')
            boomconfig.write('  case TilesLocated(InSubsystem) => up(TilesLocated(InSubsystem), site) map {\n')
            boomconfig.write('    case tp: BoomTileAttachParams => tp.copy(tileParams = tp.tileParams.copy(core = tp.tileParams.core.copy(\n')
            boomconfig.write('      bpdMaxMetaLength = {},\n'.format(system.tile.BPD.bpdMaxMetaLength))
            boomconfig.write('      globalHistoryLength = {},\n'.format(system.tile.BPD.globalHistoryLength))
            boomconfig.write('      localHistoryLength = {},\n'.format(system.tile.BPD.localHistoryLength))
            boomconfig.write('      localHistoryNSets = {},\n'.format(system.tile.BPD.localHistoryNSets))
            boomconfig.write('      branchPredictor = ((resp_in: BranchPredictionBankResponse, p: Parameters) => {\n')
            boomconfig.write('        val gshare = Module(new TageBranchPredictorBank(\n')
            boomconfig.write('          BoomTageParams(tableInfo=Seq({})))(p))\n'.format(system.tile.BPD.tage.tableInfo))
            boomconfig.write('        val btb = Module(new BTBBranchPredictorBank(BoomBTBParams(nSets={}, nWays={}, extendedNSets={}))(p))\n'.format(system.tile.BPD.btb.nSets, system.tile.BPD.btb.nWays, system.tile.BPD.btb.nSets))
            boomconfig.write('        val bim = Module(new BIMBranchPredictorBank(BoomBIMParams(nSets={}))(p))\n'.format(system.tile.BPD.bim.nSets))
            boomconfig.write('        val preds = Seq(bim, btb, gshare)\n')
            boomconfig.write('        preds.map(_.io := DontCare)\n')
            boomconfig.write('        bim.io.resp_in(0)  := resp_in\n')
            boomconfig.write('        btb.io.resp_in(0)   := bim.io.resp\n')
            boomconfig.write('        gshare.io.resp_in(0)  := btb.io.resp\n')
            boomconfig.write('        (preds, gshare.io.resp)\n')
            boomconfig.write('      })\n')
            boomconfig.write('    )))\n')
            boomconfig.write('    case other => other\n')
            boomconfig.write('  }\n')
            boomconfig.write('})\n')
        elif 'LoopG' in system.tile.BPD.name:
            boomconfig.write('\nclass With{} extends Config((site, here, up) => '.format(system.tile.BPD.name))
            boomconfig.write('{\n')
            boomconfig.write('  case TilesLocated(InSubsystem) => up(TilesLocated(InSubsystem), site) map {\n')
            boomconfig.write('    case tp: BoomTileAttachParams => tp.copy(tileParams = tp.tileParams.copy(core = tp.tileParams.core.copy(\n')
            boomconfig.write('      bpdMaxMetaLength = {},\n'.format(system.tile.BPD.bpdMaxMetaLength))
            boomconfig.write('      globalHistoryLength = {},\n'.format(system.tile.BPD.globalHistoryLength))
            boomconfig.write('      localHistoryLength = {},\n'.format(system.tile.BPD.localHistoryLength))
            boomconfig.write('      localHistoryNSets = {},\n'.format(system.tile.BPD.localHistoryNSets))
            boomconfig.write('      branchPredictor = ((resp_in: BranchPredictionBankResponse, p: Parameters) => {\n')
            boomconfig.write('        val loop = Module(new LoopBranchPredictorBank()(p))\n')
            boomconfig.write('        val gshare = Module(new TageBranchPredictorBank(\n')
            boomconfig.write('          BoomTageParams(tableInfo=Seq({})))(p))\n'.format(system.tile.BPD.tage.tableInfo))
            boomconfig.write('        val btb = Module(new BTBBranchPredictorBank(BoomBTBParams(nSets={}, nWays={}, extendedNSets={}))(p))\n'.format(system.tile.BPD.btb.nSets, system.tile.BPD.btb.nWays, system.tile.BPD.btb.nSets))
            boomconfig.write('        val bim = Module(new BIMBranchPredictorBank(BoomBIMParams(nSets={}))(p))\n'.format(system.tile.BPD.bim.nSets))
            boomconfig.write('        val preds = Seq(bim, btb, gshare, loop)\n')
            boomconfig.write('        preds.map(_.io := DontCare)\n')
            boomconfig.write('        bim.io.resp_in(0)  := resp_in\n')
            boomconfig.write('        btb.io.resp_in(0)   := bim.io.resp\n')
            boomconfig.write('        gshare.io.resp_in(0)  := btb.io.resp\n')
            boomconfig.write('        loop.io.resp_in(0)  := gshare.io.resp\n')
            boomconfig.write('        (preds, loop.io.resp)\n')
            boomconfig.write('      })\n')
            boomconfig.write('    )))\n')
            boomconfig.write('    case other => other\n')
            boomconfig.write('  }\n')
            boomconfig.write('})\n')
        else:#
            raise NameError('Unknown BPD type')
    boomconfig.close()
    #######################                 end of config-mixins                  ###########################
    #########################################################################################################
    #######################                 start of BoomConfig                   ###########################
    boomconfigpath = os.path.join(system.chipyard_path, 'generators/chipyard/src/main/scala/config/BoomConfigs.scala')
    with open (boomconfigpath, 'r') as r:
        lines = r.readlines()
    with open (boomconfigpath, 'w') as w:
        for l in lines:
            if system.name in l:
                # break
                Existflag = True
            w.write(l)
    BoomConfig = open(boomconfigpath, 'a')
    BoomConfig.write('class {} extends Config(\n'.format(system.name))
    #################################      L2 Cache  (optional)    Part II    ################################
    if system.l2cache is not None:
        BoomConfig.write('  new freechips.rocketchip.subsystem.WithNBanks({}) ++\n'.format(system.l2cache.nBanks))
        BoomConfig.write('  new chipyard.config.WithL2NWays({}) ++\n'.format(system.l2cache.nWays))
        BoomConfig.write('  new chipyard.config.WithL2NSets({}) ++\n'.format(system.l2cache.nSets))
    #################################      Tiles     ################################
    BoomConfig.write('  new boom.common.WithN{}({}) ++\n'.format(system.name, system.tile.num))
    #################################      Bus     ################################
    if system.bus.topo == 'Single':
        bus_components='s'
    elif system.bus.topo == 'Incoherent':
        bus_components='spfc'
    elif system.bus.topo == 'Coherent':
        bus_components='spfcm'

    if system.bus.frequency_sysbus is not None:
        BoomConfig.write('  new freechips.rocketchip.subsystem.WithSystemBusFrequency({}) ++\n'.format(system.bus.frequency_sysbus))
    if (system.bus.frequency_membus is not None) and ('m' in bus_components):
        BoomConfig.write('  new chipyard.config.WithMemoryBusFrequency({}) ++\n'.format(system.bus.frequency_membus))
    if (system.bus.frequency_ctrlbus is not None) and ('c' in bus_components):
        BoomConfig.write('  new chipyard.config.WithControlBusFrequency({}) ++\n'.format(system.bus.frequency_ctrlbus))
    if (system.bus.frequency_peribus is not None) and ('p' in bus_components):
        BoomConfig.write('  new chipyard.config.WithPeripheryBusFrequency({}) ++\n'.format(system.bus.frequency_peribus))
    if (system.bus.frequency_frontbus is not None) and ('f' in bus_components):
        BoomConfig.write('  new chipyard.config.WithFrontBusFrequency({}) ++\n'.format(system.bus.frequency_frontbus))

    if (system.bus.crossing_sys2mem is not None) and ('m' in bus_components):
        BoomConfig.write('  new chipyard.config.WithSbusToMbusCrossingType({}) ++\n'.format(system.bus.crossing_sys2mem))
    if (system.bus.crossing_sys2ctrl is not None) and ('c' in bus_components):
        BoomConfig.write('  new chipyard.config.WithSbusToCbusCrossingType({}) ++\n'.format(system.bus.crossing_sys2ctrl))
    if (system.bus.crossing_ctrl2peri is not None) and ('p' in bus_components):
        BoomConfig.write('  new chipyard.config.WithCbusToPbusCrossingType({}) ++\n'.format(system.bus.crossing_ctrl2peri))
    if (system.bus.crossing_front2sys is not None) and ('f' in bus_components):
        BoomConfig.write('  new chipyard.config.WithFbusToSbusCrossingType({}) ++\n'.format(system.bus.crossing_front2sys))

    if system.bus.topo == 'Ring':
        BoomConfig.write('  new testchipip.WithRingSystemBus ++)\n')
    if system.bus.topo == 'doubleRing':
        BoomConfig.write('  new testchipip.WithDoubleRingSystemBus ++)\n')
    if system.bus.topo == 'doubleRingTY':
        BoomConfig.write('  new testchipip.WithDoubleRingSystemBusTY ++)\n')
    
    BoomConfig.write('  new chipyard.config.AbstractConfig)\n')
    BoomConfig.write('\n')
    BoomConfig.close()
