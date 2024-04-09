class Bus:
    topo = 'Coherent'
    beatBytes = 8 #8 或 16
    Ring = False
    doubleRing = False
    frequency_sysbus = None    # 可选bus frequency
    frequency_membus = None
    frequency_ctrlbus = None
    frequency_peribus = None
    frequency_frontbus = None
    crossing_sys2mem = None     # NoCrossing, SynchronousCrossing(), AsynchronousCrossing() 或 RationalCrossing()
    crossing_sys2ctrl = None
    crossing_ctrl2peri = None
    crossing_front2sys = None
    def __setattr__(self, attrname, value):
        self.__dict__[attrname] = value