################################
class System:
    def __init__(self, name=None, tile=None, bus=None):
        self.name = name
        self.tile = tile
        self.bus = bus
        self.l2cache = None