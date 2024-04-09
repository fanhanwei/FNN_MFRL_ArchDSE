import os

def FixParams(system):
    rocketpath = os.path.join(system.chipyard_path, 'generators/rocket-chip/src/main/scala/subsystem/Configs.scala')
    with open (rocketpath, 'r') as r:
        lines = r.readlines()
    with open (rocketpath, 'w') as w:#清除此前写入的信息，避免程序规模不断增长
        for l in lines:
            if 'WithNcustomRockets' in l:
                break
            else:
                w.write(l)
    rocketconfig = open(rocketpath, 'a')
    rocketconfig.write('class WithNcustomRockets(n: Int, overrideIdOffset: Option[Int] = None) extends Config((site, here, up) => {\n')
    ######################################################################################################
    rocketconfig.write('  case XLen => {}\n'.format(system.tile.XLen))#Instruction Length
    rocketconfig.write('  case RocketTilesKey => {\n')
    rocketconfig.write('    val prev = up(RocketTilesKey, site)\n')
    rocketconfig.write('    val idOffset = overrideIdOffset.getOrElse(prev.size)\n')
    rocketconfig.write('    val tileParams = RocketTileParams(\n')
    rocketconfig.write('      core   = RocketCoreParams(\n')
    #################################      Core     ###############################  
    #Mode
    if system.tile.User:
        rocketconfig.write('        useUser = true,\n')
    if system.tile.Supervisor:
        rocketconfig.write('        useSupervisor = true,\n')
    #Virtual Memory (VIPT / PIPT)
    if system.tile.useVM == False:
        rocketconfig.write('        useVM = false,\n')
    #Interupts
    if system.tile.core.local_interupts > 0:
        rocketconfig.write('        nLocalInterrupts = {},\n'.format(system.tile.core.local_interupts))
    if system.tile.core.PMPs != 8:
        rocketconfig.write('        nPMPs = {},\n'.format(system.tile.core.PMPs))
    if system.tile.core.useNMI:
        rocketconfig.write('        useNMI = true,\n')
    #Debug
    if system.tile.debug == False:
        rocketconfig.write('        useDebug = false,\n')
    if system.tile.core.breakpoints != 1:
        rocketconfig.write('        nBreakpoints = {},\n'.format(system.tile.core.breakpoints))
    #Atomics
    if system.tile.core.useAtomics == False:
        rocketconfig.write('        useAtomics = false,\n')
    if system.tile.core.useAtomicsOnlyForIO == True:
        rocketconfig.write('        useAtomicsOnlyForIO = true,\n')
    #Compress & Embed
    if system.tile.core.useCompressed == False:
        rocketconfig.write('        useCompressed = false,\n')
    if system.tile.core.useRVE:
        rocketconfig.write('        useRVE = true,\n')
    #################################      Mul/Div  (optional)     ################################
    if system.tile.muldiv is not None:
        rocketconfig.write('        mulDiv = Some(MulDivParams(mulUnroll={}, divUnroll={}, mulEarlyOut={}, divEarlyOut={})),\n'.format(
            system.tile.muldiv.mul_unroll, system.tile.muldiv.div_unroll, 'true' if system.tile.muldiv.mul_earlyout else 'false', 'true' if system.tile.muldiv.div_earlyout else 'false'))
    else:
        rocketconfig.write('        mulDiv = None,\n')
    #################################      FPU  (optional)     ################################
    if system.tile.fpu is not None:
        rocketconfig.write('        fpu = Some(FPUParams(fLen={}, divSqrt={})),\n'.format(
            system.tile.fpu.flen, 'true' if system.tile.fpu.divsqrt else 'false'))
    else:
        rocketconfig.write('        fpu = None,\n')
    rocketconfig.write('        ),\n')
    #################################      Branch Predict  (optional)     ################################
    if system.tile.btb is not None:
        rocketconfig.write('      btb = Some(BTBParams(\n')
        rocketconfig.write('        nEntries = {},\n'.format(system.tile.btb.nEntries))
        rocketconfig.write('        nMatchBits = {},\n'.format(system.tile.btb.nMatchBits))
        rocketconfig.write('        nPages = {},\n'.format(system.tile.btb.nPages))
        rocketconfig.write('        nRAS = {},\n'.format(system.tile.btb.nRAS))
        rocketconfig.write('        bhtParams = Some(BHTParams(nEntries={}, counterLength={}, historyLength={}, historyBits={})),\n'.format(
            system.tile.btb.bthnEntries, system.tile.btb.bthcounterLength, system.tile.btb.bthhistoryLength, system.tile.btb.bthhistoryBits))
        rocketconfig.write('        )),\n')
    else:
        rocketconfig.write('      btb = None,\n')
    #################################     L1 Data Cache     ###############################
    rocketconfig.write('      dcache = Some(DCacheParams(\n')
    rocketconfig.write('        rowBits = site(SystemBusKey).beatBits,\n')
    rocketconfig.write('        nSets={},\n'.format(system.tile.dcache.nSets))
    rocketconfig.write('        nWays={},\n'.format(system.tile.dcache.nWays))
    rocketconfig.write('        nTLBSets={},\n'.format(system.tile.dcache.nTLBSets))
    rocketconfig.write('        nTLBWays={},\n'.format(system.tile.dcache.nTLBWays))
    rocketconfig.write('        nMSHRs={},\n'.format(system.tile.dcache.nMSHR))
    if system.tile.dcache.eccCode is not None:
        rocketconfig.write('        tagECC=Some("{}"),\n'.format(system.tile.dcache.eccCode))
        rocketconfig.write('        dataECC=Some("{}"),\n'.format(system.tile.dcache.eccCode))
    if system.tile.dcache.replacement != "random":
        rocketconfig.write('        replacementPolicy="{}",\n'.format(system.tile.dcache.replacement))
    rocketconfig.write('        blockBytes = site(CacheBlockBytes))),\n')
    #################################     L1 Inst Cache     ###############################
    rocketconfig.write('      icache = Some(ICacheParams(\n')
    rocketconfig.write('        rowBits = site(SystemBusKey).beatBits,\n')
    rocketconfig.write('        nSets={},\n'.format(system.tile.icache.nSets))
    rocketconfig.write('        nWays={},\n'.format(system.tile.icache.nWays))
    rocketconfig.write('        nTLBSets={},\n'.format(system.tile.icache.nTLBSets))
    rocketconfig.write('        nTLBWays={},\n'.format(system.tile.icache.nTLBWays))
    if system.tile.icache.eccCode is not None:
        rocketconfig.write('        tagECC=Some("{}"),\n'.format(system.tile.icache.eccCode))
        rocketconfig.write('        dataECC=Some("{}"),\n'.format(system.tile.icache.eccCode))
    if system.tile.icache.prefetch:
        rocketconfig.write('        prefetch=true,\n')
    rocketconfig.write('        blockBytes = site(CacheBlockBytes))))\n')
    ###############################################################################################
    rocketconfig.write('    List.tabulate(n)(i => tileParams.copy(hartId = i + idOffset)) ++ prev\n')
    rocketconfig.write('  }\n')
    rocketconfig.write('})\n')
    rocketconfig.close()
    #######################                 end of subsystemconfig                  ###########################
    ###########################################################################################################
    #######################                 start of RocketConfig                   ###########################
    rocketconfigpath = os.path.join(system.chipyard_path, 'generators/chipyard/src/main/scala/config/RocketConfigs.scala')
    with open (rocketconfigpath, 'r') as r:
        lines = r.readlines()
    with open (rocketconfigpath, 'w') as w:
        for l in lines:
            if system.name in l:
                break
            else:
                w.write(l)

    RocketConfig = open(rocketconfigpath, 'a')
    RocketConfig.write('class {} extends Config(\n'.format(system.name))
    #################################      L2 Cache  (optional)      ################################
    if system.l2cache is not None:
        RocketConfig.write('  new freechips.rocketchip.subsystem.WithNL2TLBWays({}) ++\n'.format(system.l2cache.ntlbWays))
        RocketConfig.write('  new freechips.rocketchip.subsystem.WithNL2TLBEntries({}) ++\n'.format(system.l2cache.ntlbEntries))
        RocketConfig.write('  new chipyard.config.WithL2NWays({}) ++\n'.format(system.l2cache.nWays))
        RocketConfig.write('  new chipyard.config.WithL2NSets({}) ++\n'.format(system.l2cache.nSets))
        RocketConfig.write('  new freechips.rocketchip.subsystem.WithNBanks({}) ++\n'.format(system.l2cache.nBanks))
    else:
        RocketConfig.write('  new freechips.rocketchip.subsystem.WithNBanks(0) ++\n')
    #################################      Tiles     ################################
    RocketConfig.write('  new freechips.rocketchip.subsystem.WithNcustomRockets({}) ++\n'.format(system.tile.num))
    #################################      Bus     ################################
    if system.bus.frequency_sysbus is not None:
        RocketConfig.write('  new chipyard.config.WithSystemBusFrequency({}) ++\n'.format(system.bus.frequency_sysbus))
    if system.bus.frequency_membus is not None:
        RocketConfig.write('  new chipyard.config.WithMemoryBusFrequency({}) ++\n'.format(system.bus.frequency_membus))
    if system.bus.frequency_ctrlbus is not None:
        RocketConfig.write('  new chipyard.config.WithControlBusFrequency({}) ++\n'.format(system.bus.frequency_ctrlbus))
    if system.bus.frequency_peribus is not None:
        RocketConfig.write('  new chipyard.config.WithPeripheryBusFrequency({}) ++\n'.format(system.bus.frequency_peribus))
    if system.bus.frequency_frontbus is not None:
        RocketConfig.write('  new chipyard.config.WithFrontBusFrequency({}) ++\n'.format(system.bus.frequency_frontbus))

    if system.bus.topo == 'Ring':
        BoomConfig.write('  new testchipip.WithRingSystemBus ++)\n')
    if system.bus.topo == 'doubleRing':
        BoomConfig.write('  new testchipip.WithDoubleRingSystemBus ++)\n')
    if system.bus.topo == 'doubleRingTY':
        BoomConfig.write('  new testchipip.WithDoubleRingSystemBusTY ++)\n')
    
    RocketConfig.write('  new chipyard.config.AbstractConfig)\n')
    RocketConfig.close()
