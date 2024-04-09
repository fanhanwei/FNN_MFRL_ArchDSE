class L1ICache:
    nSets = 64
    nWays = 4
    nTLBSets = 1
    nTLBWays = 32
    eccCode = 'parity'
    prefetch = False
    CacheBlockBytes = 64 #do not change

class L1DCache:
    nSets = 64
    nWays = 4
    nTLBSets = 1
    nTLBWays = 32
    eccCode = 'parity'
    nMSHR = 2
    replacement = "random"
    numDCacheBanks = 1