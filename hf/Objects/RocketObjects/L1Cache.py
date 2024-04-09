class L1ICache:
    nSets = 64
    nWays = 4
    nTLBSets = 1
    nTLBWays = 32
    eccCode = 'identity'
    prefetch = False

class L1DCache:
    nSets = 64
    nWays = 4
    nTLBSets = 1
    nTLBWays = 32
    eccCode = 'identity'
    nMSHR = 0
    replacement = "random"

