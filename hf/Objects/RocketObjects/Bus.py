class Bus():
    topo = 'crossbar'
    frequency_sysbus = None    # 可选bus frequency
    frequency_membus = None
    frequency_ctrlbus = None
    frequency_peribus = None
    frequency_frontbus = None
    crossing_sys2mem = None     # 可选 Asynchronous 或 Rational Crossing 
    crossing_sys2ctrl = None
    crossing_ctrl2peri = None
    crossing_front2sys = None
    def __setattr__(self, attrname, value):
            self.__dict__[attrname] = value