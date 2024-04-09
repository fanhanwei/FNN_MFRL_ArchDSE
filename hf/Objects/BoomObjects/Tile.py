class Tile:
    num = 1
    mode = 'User' #'Machine','Supervisor'
    XLen = 64
    debug = False
    muldiv = None
    fpu = None
    btb = None
    def __setattr__(self, attrname, value):
        self.__dict__[attrname] = value
        '''if attrname == "fpu":
            if value != None:
                print('  new freechips.rocketchip.subsystem.WithFPU ++', file=open('/root/pyConfig/Config.scala', 'a'))
        if attrname == "muldiv":
            if value == None:
                print('  new freechips.rocketchip.subsystem.WithoutMulDiv ++', file=open('/root/pyConfig/Config.scala', 'a'))
        if attrname == "btb":
            if value != None:
                print('  new freechips.rocketchip.subsystem.WithDefaultBtb ++', file=open('/root/pyConfig/Config.scala', 'a'))'''

