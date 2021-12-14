"""
Network architecture with loss function and optimizer
"""
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.layers.normalization import BatchNormalization
from keras.models import Input, Model
from keras.layers import (Dense, Dropout, Activation, GlobalAveragePooling3D, 
                          Conv3D, MaxPooling3D, Dropout, SpatialDropout3D)
from keras.layers import concatenate
from keras import regularizers
from nutsml import KerasNetwork

import network_decorrelate
import network_bayes

def create_optimizer(cfg):
    lr = 1/pow(10, cfg['LR'])
    optimizers = {
        0: RMSprop(lr=lr),
        1: Adam(lr=lr),
        2: Nadam(lr=lr),
        3: SGD(lr=lr, momentum=0.9, nesterov=True)
    }
    return optimizers[cfg['ALGO']]

def create_subnetwork(input, tag, cfg):

    N_FILTER = cfg['N_FILTER']
    N_CONV   = cfg['N_CONV']
    N_STRIDE = cfg['N_STRIDE']
    REG      = cfg['REG']
    BN       = cfg['BN']
    DROPOUT  = cfg['DROPOUT']

    cam_i = len(N_FILTER) - 1
    params = zip(N_FILTER, N_CONV, N_STRIDE)
    for i, (n_filter, n_conv, n_stride) in enumerate(params):
        if i == 0:
            x = Conv3D(n_filter, n_conv, strides=n_stride,
                       kernel_regularizer=regularizers.l2(REG),
                       padding='same')(input)
        else:
            x = Conv3D(n_filter, n_conv, strides=n_stride,
                       kernel_regularizer=regularizers.l2(REG),
                       padding='same')(x)

        if BN:
            x = BatchNormalization(axis=-1)(x)

        if i==cam_i:
            name = tag + '_CAM'
        else:
            name = tag + '_layer' + str(i)
        x = Activation('relu', name = name)(x)

        if DROPOUT:
            x = SpatialDropout3D(DROPOUT)(x)

    x = GlobalAveragePooling3D(name=tag+'_GAP')(x)
    return x

def create_solo_network(cfg):
    C, H, W = cfg['C'], cfg['H'], cfg['W']
    INPUTSHAPE = (H, W, C, 1)
    ROI = cfg['ROI']
    WEIGHTPATH = cfg['OUTSTEM']

    ins = Input(shape = INPUTSHAPE)
    net = create_subnetwork(ins, ROI, cfg)

    c_decorr = cfg['DECORRELATION']
    if c_decorr != 0.0:
        net = network_decorrelate.CorrelationRegularization(c_decorr)(net)

    out = Dense(1, name=ROI + 'CWGT')(net)
    out = Activation('sigmoid')(out)
    model = Model(inputs = ins, outputs = out)

    optimizer = create_optimizer(cfg)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return KerasNetwork(model, WEIGHTPATH)

########## MACONH

def create_maconh_network(cfg):
    C, H, W = cfg['C'], cfg['H'], cfg['W']
    INPUTSHAPE = (H, W, C, 1)
    WEIGHTPATH = cfg['OUTSTEM']
    N_OUT = 1

    in_mac = Input(shape = INPUTSHAPE)
    net_mac = create_subnetwork(in_mac, 'mac', cfg)

    in_onh = Input(shape = INPUTSHAPE)
    net_onh = create_subnetwork(in_onh, 'onh', cfg)

    c_decorr = cfg['DECORRELATION']
    if c_decorr == 0.0:
        out_merged = concatenate([net_mac, net_onh], name='merged_GAP')
    else:
        out_merged = concatenate([net_mac, net_onh], name='merged_GAP')
        out_merged = network_decorrelate.CorrelationRegularization(c_decorr)(out_merged)

    out = Dense(N_OUT, name= 'merged' + 'CWGT')(out_merged)
    out = Activation('sigmoid')(out)
    model = Model(inputs = [in_mac, in_onh], outputs = out)

    optimizer = create_optimizer(cfg)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return KerasNetwork(model, WEIGHTPATH)

def create_maconh_3headed_network(cfg):
    C, H, W = cfg['C'], cfg['H'], cfg['W']
    INPUTSHAPE = (H, W, C, 1)
    ROI = cfg['ROI']
    WEIGHTPATH = cfg['OUTSTEM']
    N_OUT = 1

    in_mac = Input(shape = INPUTSHAPE)
    net_mac = create_subnetwork(in_mac, 'mac', cfg)

    in_onh = Input(shape = INPUTSHAPE)
    net_onh = create_subnetwork(in_onh, 'onh', cfg)

    c_decorr = cfg['DECORRELATION']
    if c_decorr == 0.0:
        out_merged = concatenate([net_mac, net_onh], name='merged_GAP')
    else:
        out_merged = concatenate([net_mac, net_onh], name='merged_GAP')
        out_merged = network_decorrelate.CorrelationRegularization(c_decorr)(out_merged)


    if cfg['FC']:
        for i, n in enumerate(cfg['FC']):
            out_merged = Dense(n, name='FC_merged_'+str(i), activation = 'tanh')(out_merged)
            net_mac    = Dense(n, name='FC_mac_'+str(i),    activation = 'tanh')(net_mac)
            net_onh    = Dense(n, name='FC_onh_'+str(i),    activation = 'tanh')(net_onh)

    out_merged = Dense(N_OUT, name= 'merged' + 'CWGT')(out_merged)
    out_merged = Activation('sigmoid')(out_merged)

    out_mac = Dense(N_OUT, name='macCWGT')(net_mac)
    out_mac = Activation('sigmoid')(out_mac)

    out_onh = Dense(N_OUT, name='onhCWGT')(net_onh)
    out_onh = Activation('sigmoid')(out_onh)

    model = Model(inputs = [in_mac, in_onh], outputs = [out_mac, out_merged, out_onh])

    optimizer = create_optimizer(cfg)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return KerasNetwork(model, WEIGHTPATH)

############ bayes

def create_solo_bayes_network(cfg):
    C, H, W = cfg['C'], cfg['H'], cfg['W']
    INPUTSHAPE = (H, W, C, 1)
    ROI = cfg['ROI']
    WEIGHTPATH = cfg['OUTSTEM']
    lam = cfg['ALEATROPIC_WEIGHT']

    ins = Input(shape = INPUTSHAPE)
    net = create_subnetwork(ins, ROI, cfg)

    if cfg['FC']:
        for i, n in enumerate(cfg['FC']):
            net = Dense(n, name='FC_'+str(i), activation = 'tanh')(net)
    
    out_mean = Dense(1, name='CWGT_mean')(net)
    out_mean = Activation('sigmoid')(out_mean)

    out_var = Dense(1, 
                    name='CWGT_var',
                    activity_regularizer = network_bayes.reg_aleatropic(lam))(net)

    out = concatenate([out_mean, out_var])

    model = Model(inputs = ins, outputs = out)

    optimizer = create_optimizer(cfg)

    model.compile(loss = network_bayes.bnn_loss,
                  optimizer = optimizer,
                  metrics = [network_bayes.mse_metric, network_bayes.bnn_loss])

    return KerasNetwork(model, WEIGHTPATH)    

##############################################################
def create_network(cfg):
    network_type = cfg['TYPE']

    if network_type == 'solo':
        network = create_solo_network(cfg)
    elif network_type == 'maconh':
        network = create_maconh_network(cfg)
    elif network_type == 'maconh_3heads':
        network = create_maconh_3headed_network(cfg)
    elif network_type == 'solo_bayes':
        network = create_solo_bayes_network(cfg)
    else:
        print("error:", network_type)

    return network
