class Tage:
    name = 'Tage'
    bpdMaxMetaLength = 120          # BPD元数据大小上限
    globalHistoryLength = 64        # 全局历史长度
    localHistoryLength = 1          # 局部历史长度
    localHistoryNSets = 0           # 局部历史组数
    #tage                                nSets, histLen, tagSz
    class tage:
        tableInfo =   ( (  128,       2,     7),
                        (  128,       4,     7),
                        (  256,       8,     8),
                        (  256,      16,     8),
                        (  128,      32,     9),
                        (  128,      64,     9))
        uBitPeriod = 2048
    #btb
    class btb:
        nSets = 128
        nWays = 2
        offsetSz = 13
        extendedNSets = 128
    #bim
    class bim:
        nSets = 2048
    #ubtbaa
    class ubtb:
        nWays = 16
        offsetSz = 13

class Gshare:
    name = 'Gshare'
    bpdMaxMetaLength = 45          # BPD元数据大小上限
    globalHistoryLength = 16        # 全局历史长度
    localHistoryLength = 1          # 局部历史长度
    localHistoryNSets = 0           # 局部历史组数
    #tage                                nSets, histLen, tagSz
    class tage:
        tableInfo =   (256, 16, 7)
    #btb
    class btb:
        nSets = 128
        nWays = 2
        extendedNSets = 128
    #bim
    class bim:
        nSets = 2048