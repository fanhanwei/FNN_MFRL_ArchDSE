import json

def get_alg_info(pisa_out, branch_entropy, verbose=False):
    with open(pisa_out, 'r') as load_f:
        profile = json.load(load_f)
    app_name = profile['application']
    thread0 = profile['threads'][0]
    instructionMix = thread0['instructionMix']

    totalInstructions = instructionMix['instructions_analyzed']

    def inst_percentage(name):
        inst = instructionMix[name][0]
        totCount = inst['total_instructions'] / totalInstructions
        # miscCount = inst[misc_instructions] / totalInstructions
        return totCount

    F0load = inst_percentage('load_instructions')
    F0store = inst_percentage('store_instructions')
    F0mem = inst_percentage('load_instructions') + inst_percentage('store_instructions')
    F0addr = inst_percentage('address_arith_get_elem_ptr_arith_instructions') + \
            instructionMix['address_arith_alloca_arith_instructions']/totalInstructions

    F0intmul = inst_percentage('int_arith_mul_instructions')
    F0intdiv = inst_percentage('int_arith_udiv_instructions') + inst_percentage('int_arith_sdiv_instructions') + \
            inst_percentage('int_arith_urem_instructions') + inst_percentage('int_arith_srem_instructions')
    F0intOnly = inst_percentage('int_arith_add_instructions') + inst_percentage('int_arith_sub_instructions') + F0intmul + F0intdiv
    bitwise_labels = { "bitwise_and_instructions", "bitwise_or_instructions"
                ,"bitwise_xor_instructions","bitwise_shift_left_instructions"
                ,"bitwise_logical_shift_right_instructions","bitwise_arith_shift_right_instructions"}
    F0bitwise = sum([inst_percentage(n) for n in bitwise_labels])
    conversion_labels = { "conversion_trunc_instructions","conversion_zext_instructions"
                ,"conversion_sext_instructions","conversion_fptrunc_instructions"
                ,"conversion_fpext_instructions","conversion_fptoui_instructions"
                ,"conversion_fptosi_instructions","conversion_uitofp_instructions"
                ,"conversion_sitofp_instructions","conversion_inttoptr_instructions"
                ,"conversion_ptrtoint_instructions","conversion_bitcast_instructions"
                ,"conversion_address_space_cast_instructions"}
    F0conversion = sum([inst_percentage(n) for n in conversion_labels])
    F0int = F0intOnly + F0bitwise + F0conversion + inst_percentage('int_cmp_instructions')
    F0control = instructionMix['control_instructions']/totalInstructions
    F0fpmul = inst_percentage('fp_arith_mul_instructions')
    F0fpdiv = inst_percentage('fp_arith_div_instructions') + inst_percentage('fp_arith_rem_instructions')
    F0fp = inst_percentage('fp_arith_add_instructions') + inst_percentage('fp_arith_sub_instructions') \
            + inst_percentage('fp_cmp_instructions') + F0fpmul + F0fpdiv

    ILP0 = thread0['ilp'][0]['statistics']['arithmetic_mean']
    ILP0mem = thread0['ilp'][0]['statistics']['arithmetic_mean_mem']
    ILP0int = thread0['ilp'][0]['statistics']['arithmetic_mean_int']
    ILP0control = thread0['ilp'][0]['statistics']['arithmetic_mean_ctrl']
    ILP0fp = thread0['ilp'][0]['statistics']['arithmetic_mean_fp']
    ILP0WindowSize = thread0['ilp'][0]['windowSize']
    dataReuseDistance = thread0['dataReuseDistribution'][0]['statistics']['dataCDF']
    F0regreads = thread0['registerAccesses']['reads']
    F0regwrites = thread0['registerAccesses']['writes']

    entropy_list = []
    bestEntropy = 0
    bestMispredictionRate = 1
    f = open(branch_entropy, "r")
    lines = f.readlines()[1:]
    for line in lines:
        info = line.split(',')
        entropy = float(info[1])
        mispredictionRate = float(info[5])
        entropy_list.append({"bufferSize": info[0],
                            "entropy": entropy,
                            "mispredictionRate": mispredictionRate})
        if entropy > bestEntropy: bestEntropy = entropy
        if mispredictionRate < bestMispredictionRate: bestMispredictionRate = mispredictionRate
    f.close()


    profile['threads'][0]['branchEntropy'] = {
                'entropies': entropy_list,
                'branches': 0,
                'bestEntropy': bestEntropy,
                'bestMispredictionRate': bestMispredictionRate
                }
    # with open(args.p + 'with_br', 'w') as write_f:
    # 	json.dump(profile, write_f, indent=4, ensure_ascii=False)

    if verbose: 
        print('totalInstructions', totalInstructions)
        print('F0int', F0int)
        print('F0intmul', F0intmul)
        print('F0intdiv', F0intdiv)
        print('F0addr', F0addr)
        print('F0intOnly', F0intOnly)

        print('F0mem', F0mem)
        print('F0control', F0control)

        print('F0fpmul', F0fpmul)
        print('F0fpdiv', F0fpdiv)
        print('F0fp', F0fp)

        print('ILP0', ILP0)
        print('ILP0mem', ILP0mem)
        print('ILP0int', ILP0int)
        print('ILP0control', ILP0control)
        print('ILP0fp', ILP0fp)
        print('ILP0WindowSize', ILP0WindowSize)
        print('dataReuseDistance', dataReuseDistance)
        print('bestMispredictionRate', bestMispredictionRate)
    
    return_dict = {
        'alg_name': app_name,
        'F0int': F0int,
        'F0intOnly': F0intOnly,
        'F0intmul': F0intmul,
        'F0intdiv': F0intdiv,
        'F0addr': F0addr,
        'F0load': F0load,
        'F0store': F0store,
        'F0mem': F0mem,
        'F0control': F0control,
        'F0fp': F0fp,
        'F0fpmul': F0fpmul,
        'F0fpdiv': F0fpdiv,
        'ILP0': ILP0,
        'ILP0mem': ILP0mem,
        'ILP0int': ILP0int,
        'ILP0control': ILP0control,
        'ILP0fp': ILP0fp,
        'ILP0WindowSize': ILP0WindowSize,
        'dataReuseDistance': dataReuseDistance,
        'F0regreads': F0regreads,
        'F0regwrites': F0regwrites,
        'bestMispredictionRate': bestMispredictionRate,
        'totalInstructions': totalInstructions
    }

    return return_dict

if __name__ == '__main__':
    import argparse                          # run time flag handler
    arg_parser = argparse.ArgumentParser(description='get info from pisa out')
    arg_parser.add_argument('-p', type=str, help='the pisa profile' )
    arg_parser.add_argument('-b', type=str, help='the branch analysis results')
    arg_parser.add_argument('-v', dest='verbose', action='store_true', help='print detailed info')
    args = arg_parser.parse_args()
    alg = get_alg_info(args.p, args.b, args.v)