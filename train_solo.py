"""
Network training
"""
import os
import sys
import argparse
import csv
import re

import numpy as np
import sklearn.metrics as sm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import nutsml.config

from common import *
from network import create_network

from datetime import datetime, timedelta

@nut_sink
def CollectMetrics(metrices):
    return metrices >> Unzip() >> Map(Mean()) >> Collect()

@nut_processor
def ReadVolumesOneBranch(samples, cfg):
    return samples >> ReadSingleCube3d(cfg) >> MakeSingleEnFace()

@nut_processor
def ReadVolumes(samples):
    return samples >> read_cube >> make_enface

def rmse(tars, preds):
    return np.sqrt(((preds - tars) ** 2).mean())    

def rmae(tars, preds):
    return np.abs((preds - tars)).mean()

# this will break if the network is predicting more than 1 parameter
def performance(network, samples, cfg):  
    if cfg['Y']=='vfi':
        idx = 0
    elif cfg['Y']=='md':
        idx = 1
    else:
        print("error :", cfg['Y'])
    descale = lambda pred: scale_vft(pred, idx, False)

    build_pred_batch = build_pred_batcher(cfg)

    preds = (samples >> 
             ReadSingleCube3d(cfg) >> 
             MakeSingleEnFace() >> 
             build_pred_batch >> 
             network.predict() >> 
             Collect())

    corrs, rhos, rmses, rmaes = [], [], [], []

    for col in range(1):
        ts = samples >> Get(2) >> Map(descale) >> Get(col) >> Collect()
        
        if cfg['TYPE'] in ['solo_bayes']:
            # for bayesian networks, only the first element of the output vector is the predicted value
            ps = preds >> Get(0) >> Map(descale) >> Collect()
        else:
            ps = preds >> Map(descale) >> Get(col) >> Collect()
        
        ts, ps  = np.array(ts), np.array(ps)
        r, p = pearsonr(ts, ps)
        rho, p_rho = spearmanr(ts, ps)
        corrs.append(round(abs(r),2))
        rhos.append(round(abs(rho), 2))
        rmses.append(rmse(ts, ps))
        rmaes.append(rmae(ts, ps))

    return corrs, rhos, rmses, rmaes


# want: uid, dx, train/val/test, truth, predicted
def predict(fold, cfg, out_dir):
    network = create_network(cfg)
    network.load_weights(cfg['WEIGHTS_STEM'] + '_' + str(fold) + '.h5')
    
    train_samples, val_samples, test_samples = read_samples(fold, cfg)
    print('#samples', len(train_samples), len(val_samples), len(test_samples))

    build_pred_batch = build_pred_batcher(cfg)
    
    if cfg['Y']=='vfi':
        idx = 0
    elif cfg['Y']=='md':
        idx = 1
    else:
        print("error ", cfg['Y'])

    descale = lambda pred: scale_vft(pred, idx, False)

    res = []
    for (samples, label) in list(zip([train_samples, val_samples, test_samples], ["train", "val", "test"])):
        
        uid_lst   = samples >> Get(0) >> Collect()
        dx_lst    = samples >> Get(1) >> Collect()
        fold_lst  = [fold] * len(samples) 
        label_lst = [label] * len(samples)
        val_lst   = samples >> Get(2) >> Map(descale) >> Collect()
        val_lst   = val_lst >> Map(lambda x: x.tolist()) >> Collect()

        preds = (samples >>
                 ReadSingleCube3d(cfg) >>
                 MakeSingleEnFace() >>
                 build_pred_batch >>
                 network.predict() >>
                 Collect())
        ps = preds >> Map(descale) >> Collect()
        ps = ps >> Map(lambda x: x.tolist()) >> Collect()

        zz = list(zip(uid_lst, fold_lst, dx_lst, label_lst, val_lst, ps))

        # quick and dirty flatten
        res0 = [[item[0], item[1], item[2], item[3]] + item[4] + item[5] for item in zz]
        res = res + res0
    return res

def train(fold, cfg, out_dir):
    network = create_network(cfg)

    train_samples, val_samples, test_samples = read_samples(fold, cfg)
    print('#samples', len(train_samples), len(val_samples), len(test_samples))

    best_v_loss = 100

    log_train_filename = out_dir + '/log_train_' + cfg['OUTSTEM'] + '.csv'
    log_save_filename  = out_dir + '/log_save_'  + cfg['OUTSTEM'] + '.csv'
    weights_filename   = out_dir + '/best_weights_' + cfg['OUTSTEM']

    plot_epoch = PlotLines(list(range(4)), 
                           layout=(4, None), 
                           figsize=(22, 16), 
                           filepath= out_dir + '/epoch_plot.png')

    time0 = datetime.now()
    print("time=", str(time0))

    with open(log_train_filename, 'a') as log_train_fp, open(log_save_filename, 'a') as log_save_fp:
        log_train = csv.writer(log_train_fp)
        log_save  = csv.writer(log_save_fp)

        build_batch = build_batcher(cfg)

        for epoch in range(cfg['N_EPOCHS']):
            print('EPOCH:', epoch)
            print("time=", str(datetime.now()))

            t_loss, t_mse = (train_samples >> 
                             ReadVolumesOneBranch(cfg) >>
                             Shuffle(100) >> 
                             build_batch >> 
                             network.train() >> 
                             CollectMetrics())
            print('train : %.5f %.5f' % (t_loss, t_mse))
            sys.stdout.flush()

            v_loss, v_mse = (val_samples >> 
                             ReadVolumesOneBranch(cfg) >> 
                             build_batch >> 
                             network.validate() >> 
                             CollectMetrics())
            print('val   : %.5f  %.5f' % (v_loss, v_mse)) 
            sys.stdout.flush()

            train_log_stats = [fold, epoch, t_loss, v_loss]

            if epoch % 5 == 0:
                corrs, rhos, rmses, rmaes = performance(network, test_samples, cfg)
                print('test  :', corrs, rhos, rmses, rmaes)
                sys.stdout.flush()
                train_log_stats = train_log_stats + corrs + rhos, rmses + rmaes

            log_train.writerow(train_log_stats)
            log_train_fp.flush()
        
            if v_loss < best_v_loss:
                best_v_loss = v_loss
                print("**saving weights ", fold, epoch)
                sys.stdout.flush()
                network.save_weights(weights_filename + '_' + str(fold) + ".h5")
                log_save.writerow([fold, epoch, best_v_loss])
                log_save_fp.flush()

    print("===============")
    time1 = datetime.now()
    print("time=", str(time0))
    print("duration=", str(time1-time0))


                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config_filename', help='name of config file')
    args = parser.parse_args()

    config_file = args.config_file
    cfg = nutsml.config.load_config(args.config_file)

    if cfg["DEVICE"] == "DLaaS":
        data_dir = os.environ["DATA_DIR"]
        out_dir  = os.environ["RESULT_DIR"]
        cfg['IMG_PATH'] = data_dir + '/'
        cfg['WEIGHTS_STEM'] = data_dir + '/' + cfg['WEIGHTS_STEM']
    else:
        out_dir = "."
        init_GPU(cfg) 

    if cfg['OUTSTEM'] == 'DEFAULT':

        # assumes that the config file is called config_BLABLABLA.yaml
        # if it is not, use the whole filename as stem
        p = re.compile('config_([a-zA-Z0-9._]+).yaml')
        mobj = p.match(config_file)

        if mobj:
            cfg['OUTSTEM'] = mobj.group(1)
        else:
            cfg['OUTSTEM'] = config_file.split('.')[0]

    N_FOLDS = cfg['N_FOLDS']
    if isinstance(N_FOLDS, list):
        n0 = N_FOLDS[0]
        n1 = N_FOLDS[1]
    else:
        n0 = 0
        n1 = N_FOLDS

    try:
        predictP = cfg['PREDICT']
    except:
        predictP = False

    if predictP:
        res = []
        for fold in range(n0, n1):
            print('predicting FOLD ', fold, ' of', [n0, n1])
            res0 = predict(fold, cfg, out_dir)
            res = res + res0
        
        with open(out_dir + '/' + 'prediction-' + cfg['OUTSTEM'] + '.csv', 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(res)

    else:
        for fold in range(n0, n1):
            print('training FOLD ', fold, ' of', [n0, n1])
            train(fold, cfg, out_dir)

