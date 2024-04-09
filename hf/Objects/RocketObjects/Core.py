class Core:
    name = 'rocket'
    local_interupts = 0
    useNMI = False
    PMPs = 8
    breakpoints = 1
    useRVE = False
    useCompressed = True
    useAtomics = True
    useAtomicsOnlyForIO = False

    def __setattr__(self, attrname, value):
        self.__dict__[attrname] = value
