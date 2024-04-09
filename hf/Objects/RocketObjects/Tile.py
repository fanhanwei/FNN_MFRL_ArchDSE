class Tile:
    num = 1
    mode = 'Machine' #'Machine','Supervisor'
    XLen = 64
    useVM = True
    debug = True
    muldiv = None
    fpu = None
    btb = None
    def __setattr__(self, attrname, value):
        self.__dict__[attrname] = value
