from __future__ import print_function

import os
from nutsflow.common import StableRandom

import numpy as np
import os.path as osp
import scipy as sp

from nutsflow import *
from nutsml import *

def init_GPU(cfg):

    DEVICE = cfg['DEVICE']

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress tensorflow warnings
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE  # 0,1,... for GPU or -1 for CPU

@nut_function
def ReadSingleCube3d(sample, cfg):
    # sample is [[uid, dx, v] ...]
    # return    [[cube, v...] ...] where v is a numpy vector of test numbers
    uid = sample[0]
    cube_filename = cfg['PATH_PREFIX'] + cfg['IMG_PATH'] + cfg['ROI'].lower() + '/' + uid + '.npy'
    cube = np.load(cube_filename)
    if '-OD' in uid:
        cube = np.flip(cube, 2)

    if cfg['TYPE']=='solo':
        res = [cube] + list(sample[2:])
    elif cfg['TYPE']=='solo_bayes':
        # if this is for solo_bayes, the output is duplicated. This is just a hack to match to the output side of the network 
        # The bayes network output sigmas, so the size of the output size is doubled.
        val = sample[2][0]
        res = [cube, np.array([val, val])]
    else:
        res = None
    return res

@nut_function
def MakeSingleEnFace(sample):
    cube = sample[0]
    cube = np.transpose(cube, (1, 0, 2))
    cube = np.expand_dims(cube, 3)
    return [cube] + sample[1:]

@nut_function
def ReadCube3d(sample, cfg):
    # sample is [[uid, dx, v] ...]
    # return    [[cube, v...] ...] where v is a numpt vector of test numbers
    uid = sample[0]
    stem = cfg['PATH_PREFIX'] + cfg['IMG_PATH']
    cube1 = np.load(osp.join(stem, 'mac', uid + '.npy'))
    cube2 = np.load(osp.join(stem, 'onh', uid + '.npy'))
    if '-OD' in uid:
        cube1 = np.flip(cube1, 2)
        cube2 = np.flip(cube2, 2)
    
    if cfg['TYPE'] in ['maconh', 'maconh_3heads']:
        res = [(cube1, cube2)] + list(sample[2:])
    elif cfg['TYPE'] in ['solo_bayes', 'twin']:
        # the output is duplicated. This is a hack. 
        res = [(cube1, cube2)] + list(sample[2:]) + list(sample[2:]) 
    return res

@nut_function
def MakeEnFace(sample):
    (cube1, cube2) = sample[0]
    cube1 = np.transpose(cube1, (1, 0, 2))
    cube1 = np.expand_dims(cube1, 3)
    cube2 = np.transpose(cube2, (1, 0, 2))
    cube2 = np.expand_dims(cube2, 3)
    return [cube1, cube2] + sample[1:]

@nut_processor
def StratifyVFI(samples, fold, bins=4):
    vfi = lambda sample: sample[2]
    rand = StableRandom(fold)
    values = samples >> Map(vfi) >> Collect()
    hist, _ = np.histogram(values, bins=bins, density=False)
    hist = hist + 1
    probs = float(min(hist)) / hist
    for sample in samples:
        idx = int(vfi(sample) * bins)
        idx = min(bins - 1, max(0, idx))
        if rand.random() < probs[idx]:
            yield sample

# this is a very unfortunate piece of code, causing many problems.
# but I dont want to change it
# vfts can be a numpy array of one element, then the result is a numpy array of one element
# vfts can also be a scalar. Then the result is also a scalar 
def scale_vft(vfts, idx, down=True):
    #                 vfi, md, psd, ght
    v_min = np.array([  0., -33.,  0., 1.])
    v_max = np.array([100.,   3., 20., 6.])
    mi, ra = v_min[idx], (v_max - v_min)[idx]
    vfts = np.array(vfts) 
    if down:
        return 0.1 + (((vfts - mi) / ra) / 1.25)
    else: 
        return (((vfts - 0.1) * 1.25) * ra) + mi
    
# originally read_samples has a scale option, set to True by default
# I decided to take away this option, since I dont think it's ever used.
# also, the original code discard DX. I decided to keep it.
# return [[uid, dx, v]...] where v is a numpy vector of test numbers
# cfg['DATASET'] can specify a csv file. If this is not set, use vft.csv
# cfg['SPLIT'] can be used to stop splitting. Default is to split
# cfg['SUBJECTS'] filters by patients
def read_samples(fold, cfg):
    rand = StableRandom(fold)
    same_sid = lambda s: s[0].split('-')[0]

    try:
        splitQ = cfg['SPLIT']
    except:
        splitQ = True

    try:
        subjects = cfg['SUBJECTS']
        splitQ = False
    except:
        subjects = []

    split_data = SplitRandom((0.8, 0.1, 0.1), constraint=same_sid, rand=rand)

    dx_filter = cfg['DX']
    filter_dx = FilterCol(1, lambda dx: not dx_filter or dx in dx_filter)

    if not(subjects==[]):
        # in subject modes, we don't filter by DX
        filter_dx = NOP(filter_dx)

    # keep uid and dx (the first two element) , then transform the rest
    # if training a 3-headed network, replicate the outputs for the three heads
    def f(lst):
        if cfg['Y']=='vfi':
            idx = 0
        elif cfg['Y']=='md':
            idx = 1
        else:
            print("error ", cfg['Y'])
        return scale_vft(lst, idx, down=True)

    if cfg['TYPE']=='maconh_3heads':
        reformat = Map(lambda s: (s[0], s[1], f(s[2:]), f(s[2:]), f(s[2:])))
    else:
        reformat = Map(lambda s: (s[0], s[1], f(s[2:])))
    
    cols = ['uid', 'dx', cfg['Y']]

    try:
        dataset_csv = cfg['DATASET']
    except:
        dataset_csv = 'vft.csv'

    data = ReadPandas(dataset_csv, columns=cols)

    if not(subjects==[]):
        # "SUBJECTS" mode
        uid = data.dataframe['uid']
        uid = uid.apply(lambda x: (x[:6] + x[-3:] in subjects))
        data.dataframe = data.dataframe[uid]
    
    if not (cfg['LIMIT_N']==False):
        data2 = data >> filter_dx >> Take(cfg['LIMIT_N']) >> Collect()
    else:
        data2 = data >> filter_dx >> Collect()

    try:
        stratifyP = cfg['STRATIFY']
    except:
        stratifyP = False

    if splitQ:
        if stratifyP:
            data2 = data2 >> StratifyVFI(fold) >> reformat >> split_data >> Collect()
        else:
            data2 = data2 >> reformat >> split_data >> Collect()
    else:
        if stratifyP:
            data2 = data2 >> StratifyVFI(fold) >> reformat >> Collect()
        else:
            data2 = data2 >> reformat >> Collect()
        data2 = data2, [], []

    return data2

read_cube = ReadCube3d()

make_enface = MakeEnFace()

def build_batcher(cfg):
    network_type = cfg['TYPE']
    if network_type in ['solo', 'solo_bayes']:
        build_batch = (BuildBatch(cfg['BATCH_SIZE'], prefetch=False)
                       .input(0, 'tensor', 'float32')
                       .output(1, 'vector', 'float32'))
    elif network_type in ['merge_solo_latish', 'merge_solo_late', 'maconh', 'twin']:
        build_batch = (BuildBatch(cfg['BATCH_SIZE'], prefetch=False)
                       .input(0, 'tensor', 'float32')
                       .input(1, 'tensor', 'float32')
                       .output(2, 'vector', 'float32'))
    elif network_type == 'maconh_3heads':
        build_batch = (BuildBatch(cfg['BATCH_SIZE'], prefetch=False)
                       .input(0, 'tensor', 'float32')
                       .input(1, 'tensor', 'float32')
                       .output(2, 'vector', 'float32')
                       .output(3, 'vector', 'float32')
                       .output(4, 'vector', 'float32'))
    else:
        print("err:", network_type)

    return build_batch

def build_pred_batcher(cfg):
    network_type = cfg['TYPE']
    if network_type in ['solo', 'solo_bayes']:
        build_pred_batch = (BuildBatch(cfg['BATCH_SIZE'], prefetch=False)
                            .input(0, 'tensor', 'float32'))
    elif network_type in ['merge_solo_latish', 'merge_solo_late', 'maconh', 'maconh_3heads']:
        build_pred_batch = (BuildBatch(cfg['BATCH_SIZE'], prefetch=False)
                            .input(0, 'tensor', 'float32')
                            .input(1, 'tensor', 'float32'))
    else:
        print("err:", network_type)

    return build_pred_batch

