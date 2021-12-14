"""
Network training
Take two cubes
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

@nut_sink
def CollectMetrics(metrices):
    return metrices >> Unzip() >> Map(Mean()) >> Collect()

@nut_processor
def ReadVolumesOneBranch(samples, cfg):
    return samples >> ReadSingleCube3d(cfg) >> MakeSingleEnFace()

@nut_processor
def ReadVolumes(samples, cfg):
    return samples >> ReadCube3d(cfg) >> MakeEnFace()

def rmse(tars, preds):
    return np.sqrt(((preds - tars) ** 2).mean())    

def rmae(tars, preds):
    return np.abs((preds - tars)).mean()

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
             ReadCube3d(cfg) >> 
             MakeEnFace() >> 
             build_pred_batch >> 
             network.predict(flatten=False) >> 
             Collect())

    preds_mac    = preds >> Map(lambda x: x[0]) >> Collect()
    preds_maconh = preds >> Map(lambda x: x[1]) >> Collect()
    preds_onh    = preds >> Map(lambda x: x[2]) >> Collect()

    preds_mac    = np.vstack(preds_mac).tolist()
    preds_maconh = np.vstack(preds_maconh).tolist()
    preds_onh    = np.vstack(preds_onh).tolist()

    corrs_mac,    rhos_mac,    rmses_mac,    rmaes_mac    = [], [], [], []
    corrs_maconh, rhos_maconh, rmses_maconh, rmaes_maconh = [], [], [], []
    corrs_onh,    rhos_onh,    rmses_onh,    rmaes_onh    = [], [], [], []

    for col in range(1):
        ts = samples >> Get(2) >> Map(descale) >> Get(col) >> Collect()

        ps_mac     = preds_mac    >> Map(descale) >> Get(col) >> Collect()
        ps_maconh  = preds_maconh >> Map(descale) >> Get(col) >> Collect()
        ps_onh     = preds_onh    >> Map(descale) >> Get(col) >> Collect()

        ts         = np.array(ts)
        ps_mac     = np.array(ps_mac)
        ps_maconh  = np.array(ps_maconh)
        ps_onh     = np.array(ps_onh)

        r_mac, p_mac = pearsonr(ts, ps_mac)
        corrs_mac.append(round(abs(r_mac),2))
        r_maconh, p_maconh = pearsonr(ts, ps_maconh)
        corrs_maconh.append(round(abs(r_maconh),2))
        r_onh, p_onh = pearsonr(ts, ps_onh)
        corrs_onh.append(round(abs(r_onh),2))

        rho_mac,    p_rho_mac    = spearmanr(ts, ps_mac)
        rho_maconh, p_rho_maconh = spearmanr(ts, ps_maconh)
        rho_onh,    p_rho_onh    = spearmanr(ts, ps_onh)
        rhos_mac.append(   round(abs(rho_mac),   2))
        rhos_maconh.append(round(abs(rho_maconh),2))
        rhos_onh.append(   round(abs(rho_onh),   2))
        
        rmses_mac.append(rmse(ts, ps_mac))
        rmses_maconh.append(rmse(ts, ps_maconh))
        rmses_onh.append(rmse(ts, ps_onh))

        rmaes_mac.append(rmae(ts, ps_mac))
        rmaes_maconh.append(rmae(ts, ps_maconh))
        rmaes_onh.append(rmae(ts, ps_onh))

    return (corrs_mac, corrs_maconh, corrs_onh, 
            rhos_mac,  rhos_maconh,  rhos_onh,
            rmses_mac, rmses_maconh, rmses_onh,
            rmaes_mac, rmaes_maconh, rmaes_onh)

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
                 ReadCube3d(cfg) >>
                 MakeEnFace() >>
                 build_pred_batch >>
                 network.predict(flatten=False) >>
                 Collect())

        preds_mac    = preds >> Map(lambda x: x[0]) >> Map(descale) >> Collect()
        preds_maconh = preds >> Map(lambda x: x[1]) >> Map(descale) >> Collect()
        preds_onh    = preds >> Map(lambda x: x[2]) >> Map(descale) >> Collect()

        preds_mac    = np.vstack(preds_mac).tolist()
        preds_maconh = np.vstack(preds_maconh).tolist()
        preds_onh    = np.vstack(preds_onh).tolist()

        #             0        1         2       3          4        5          6             7
        zz = list(zip(uid_lst, fold_lst, dx_lst, label_lst, val_lst, preds_mac, preds_maconh, preds_onh))

        # quick and dirty flatten
        res0 = [[item[0], item[1], item[2], item[3]] + item[4] + item[5] +item[6] + item[7] for item in zz]
        res = res + res0
    return res

def train(fold, cfg, out_dir):
    network = create_network(cfg)
    network.print_network()

    train_samples, val_samples, test_samples = read_samples(fold, cfg)
    print('#samples', len(train_samples), len(val_samples), len(test_samples))
    print(train_samples[0])

    best_v_loss = 1000

    log_train_filename = out_dir + '/log_train_' + cfg['OUTSTEM'] + '.csv'
    log_save_filename  = out_dir + '/log_save_'  + cfg['OUTSTEM'] + '.csv'
    weights_filename   = out_dir + '/best_weights_' + cfg['OUTSTEM']

    plot_epoch = PlotLines(list(range(4)), 
                           layout=(4, None), 
                           figsize=(22, 16), 
                           filepath= out_dir + '/epoch_plot.png')

    with open(log_train_filename, 'a') as log_train_fp, open(log_save_filename, 'a') as log_save_fp:
        log_train = csv.writer(log_train_fp)
        log_save  = csv.writer(log_save_fp)

        build_batch = build_batcher(cfg)

        for epoch in range(cfg['N_EPOCHS']):
            print('EPOCH:', epoch)

            zz = (train_samples >> 
                  ReadVolumes(cfg) >>
                  Shuffle(100) >> 
                  build_batch >> 
                  network.train() >> 
                  CollectMetrics())

            t_loss_total  = zz[0]
            t_loss_mac    = zz[1]
            t_loss_maconh = zz[2]
            t_loss_onh    = zz[3]
            print('train : %.5f %.5f %.5f %.5f' % (t_loss_total, t_loss_mac, t_loss_maconh, t_loss_onh))
            sys.stdout.flush()

            zz = (val_samples >> 
                  ReadVolumes(cfg) >> 
                  build_batch >> 
                  network.validate() >> 
                  CollectMetrics())
            v_loss_total  = zz[0]
            v_loss_mac    = zz[1]
            v_loss_maconh = zz[2]
            v_loss_onh    = zz[3]
            print('val   : %.5f %.5f %.5f %.5f' % (v_loss_total, v_loss_mac, v_loss_maconh, v_loss_onh))
            sys.stdout.flush()

            train_log_stats = [fold, epoch, t_loss_total, t_loss_mac, t_loss_maconh, t_loss_onh, v_loss_total, v_loss_mac, v_loss_maconh, v_loss_onh]

            if epoch % 5 == 0:
                corrs_mac, corrs_maconh, corrs_onh, rhos_mac, rhos_maconh, rhos_onh, rmses_mac, rmses_maconh, rmses_onh, rmaes_mac, rmaes_maconh, rmaes_onh = performance(network, test_samples, cfg)
                print('test mac   :', corrs_mac,    rhos_mac,    rmses_mac,    rmaes_mac)
                print('test maconh:', corrs_maconh, rhos_maconh, rmses_maconh, rmaes_maconh)
                print('test onh   :', corrs_onh,    rhos_onh,    rmses_onh,    rmaes_onh)
                sys.stdout.flush()
                train_log_stats = train_log_stats + corrs_mac + rhos_mac + rmses_mac + rmaes_mac + corrs_maconh + rhos_maconh + rmses_maconh + rmaes_maconh + corrs_onh + rhos_onh + rmses_onh + rmaes_onh

            log_train.writerow(train_log_stats)
            log_train_fp.flush()
        
            if v_loss_total < best_v_loss:
                best_v_loss = v_loss_total
                print("**saving weights ", fold, epoch)
                sys.stdout.flush()
                network.save_weights(weights_filename + '_' + str(fold) + ".h5")
                log_save.writerow([fold, epoch, best_v_loss])
                log_save_fp.flush()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config_filename', help='name of config file')
    args = parser.parse_args()

    config_file = args.config_file
    cfg = nutsml.config.load_config(args.config_file)

    if cfg["DEVICE"] == "DLaaS":
        data_dir = os.environ["DATA_DIR"]
        out_dir  = os.environ["RESULT_DIR"]

        try:
            dlaas_path = cfg['DLAAS_PATH']
        except:
            dlaas_path = ''

        cfg['IMG_PATH'] = data_dir + '/' + dlaas_path + '/'
        cfg['WEIGHTS_STEM'] = data_dir + '/' +cfg['WEIGHTS_STEM']
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
