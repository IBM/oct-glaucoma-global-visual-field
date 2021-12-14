import argparse
import scipy as sp
import nibabel as nib
import csv

import keras.backend as K

import nutsml.config
import nutsml.imageutil as ni

from network import create_network
from common import *

def code_split(str):
    if str=='train':
        code = 0
    elif str=='val':
        code = 1
    elif str=='test':
        code = 2
    else:
        print("code_split error")
        code = -1
    return code

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def resize_cube(cube, shape):
    """Return resized cube with the define shape"""
    zoom = [float(x) / y for x, y in zip(shape, cube.shape)]
    resized = sp.ndimage.zoom(cube, zoom)
    return resized

@nut_processor
def ReadVolumesOneBranch(samples, cfg):
    return samples >> ReadSingleCube3d(cfg) >> MakeSingleEnFace()

@nut_processor
def ReadVolumes(samples, cfg):
    return samples >> ReadCube3d(cfg) >> MakeEnFace()

# shape:
#     solo: [1, N]
def get_weights(network, cfg, bias=False):
    model = network.model

    if cfg['TYPE'] == 'solo':
        # assuming that there is only one output unit
        if cfg['ROI'] == 'MAC':
            w1 = model.get_layer('MACCWGT').get_weights()[0]
            w2 = model.get_layer('MACCWGT').get_weights()[1]
            if bias:
                w = np.concatenate((w1.flatten(), w2.flatten()))
            else:
                w = w1
        elif cfg['ROI'] == 'ONH':
            w1 = model.get_layer('ONHCWGT').get_weights()[0]
            w2 = model.get_layer('ONHCWGT').get_weights()[1]
            if bias:
                w = np.concatenate((w1.flatten(), w2.flatten()))
            else:
                w = w1
        else:
            print('error')
        w = w.reshape([1, w.size])
    elif cfg['TYPE'] == 'maconh_3heads':
        # assuming that there is only one output unit

        if cfg['ROI'] == 'MAC':
            w1 = model.get_layer('macCWGT').get_weights()[0]
            w2 = model.get_layer('macCWGT').get_weights()[1]
            if bias:
                w = w1 + w2
            else:
                w = w1
            w = w.reshape([1, w.size])
        elif cfg['ROI'] == 'ONH':
            w1 = model.get_layer('onhCWGT').get_weights()[0]
            w2 = model.get_layer('onhCWGT').get_weights()[1]
            if bias:
                w = w1 + w2
            else:
                w = w1
            w = w.reshape([1, w.size])
        elif cfg['ROI'] == 'MACONH':
            w1 = model.get_layer('mergedCWGT').get_weights()[0]
            w2 = model.get_layer('mergedCWGT').get_weights()[1]
            if bias:
                w = np.concatenate((w1.flatten(), w2.flatten()))
            else:
                w = w1
            w = w.reshape([1, w.size])
        elif cfg['ROI'] in ['MAConh', 'macONH']:
            # this doensnt support bias, because it's only used for generating CAMs, not for exporting weights
            w = model.get_layer('mergedCWGT').get_weights()[0]
            w = w.reshape([1, w.size])
            half_size = int(w.size/2)
            if cfg['ROI'] == 'MAConh':
                w = w[0, :half_size]
            elif cfg['ROI'] == 'macONH':
                w = w[0, half_size:]
            else:
                print("error")
            w = w.reshape([1, half_size])
        else:
            print("error")
    else:
        print('error')
    
    return w

@nut_function
def get_GAP_activations(batch, network, cfg):
    # batch is [tensor] or [tensor, tensor]

    model = network.model
    if cfg['TYPE']=='solo':
        if cfg['ROI'] == 'MAC':
            roi = 'MAC'
        elif cfg['ROI'] == 'ONH':
            roi = 'ONH'
    elif cfg['TYPE'] == 'maconh_3heads':
        if cfg['ROI'] in ['MAC', 'MAConh']:
            roi = 'mac'
        elif cfg['ROI'] in ['ONH', 'macONH']:
            roi = 'onh'
        elif cfg['ROI'] == 'MACONH':
            roi ='merged'

    if cfg['TYPE'] == 'solo':
        f = K.function([model.input, K.learning_phase()], 
                       [model.get_layer(roi + '_GAP').output])

        activations = f(batch + [0])[0]
    else:
        f = K.function(model.input + [K.learning_phase()], 
                       [model.get_layer(roi + '_GAP').output])

        activations = f(batch + [0])[0]

    return activations

@nut_function
def get_CAM_activations(batch, network, cfg):
    # batch is [tensor] or [tensor, tensor]

    model = network.model
    if cfg['TYPE']=='solo':
        if cfg['ROI'] == 'MAC':
            roi = 'MAC'
        elif cfg['ROI'] == 'ONH':
            roi = 'ONH'
    elif cfg['TYPE'] == 'maconh_3heads':
        if cfg['ROI'] in ['MAC', 'MAConh']:
            roi = 'mac'
        elif cfg['ROI'] in ['ONH', 'macONH']:
            roi = 'onh'

    if cfg['TYPE'] == 'solo':
        f = K.function([model.input, K.learning_phase()], 
                       [model.get_layer(roi + '_CAM').output])

        activations = f(batch + [0])[0]
    else:
        f = K.function(model.input + [K.learning_phase()], 
                       [model.get_layer(roi + '_CAM').output])

        activations = f(batch + [0])[0]

    return activations

@nut_function
def CAM3D(act_batch, w):
    
    (w_pos, w_neg) = decompose_weights(w)

    # w:             (1, 32)
    #                 i  j   k   l   m
    # act_batch:     (_, 64, 32, 32, 32)
    # cam:           (_, 64, 32, 32)

    cam     = np.einsum('ijklm,nm->ijkl', act_batch, w)
    cam_pos = np.einsum('ijklm,nm->ijkl', act_batch, w_pos)
    cam_neg = np.einsum('ijklm,nm->ijkl', act_batch, w_neg)
    return (cam, cam_pos, cam_neg)

def scale_cam(cam, scale='automatic'):
    if scale == 'automatic':
        mn = np.min(cam)
        mx = np.max(cam)
    else:
        mn = scale[0]
        mx = scale[1]

    cam2 = ni.rerange(cam, mn, mx, 0, 255, 'uint8')
    cam2 = resize_cube(cam2, [128, 64, 64])
    return cam2

def export_nifty(cam, filestem, lefteyeQ=True):
    if lefteyeQ:
        # SAR
        tr = np.array([[ 0.0,  0.0, -1.0, 0.0],
                       [ 0.0, -1.0,  0.0, 0.0],
                       [-1.0,  0.0,  0.0, 0.0],
                       [ 0.0,  0.0,  0.0, 1.0]])
    else:
        # SAL
        tr = np.array([[ 0.0,  0.0,  1.0, 0.0],
                       [ 0.0, -1.0,  0.0, 0.0],
                       [-1.0,  0.0,  0.0, 0.0],
                       [ 0.0,  0.0,  0.0, 1.0]])
    img = nib.Nifti1Image(cam, tr)
    nib.save(img, filestem + '.nii')

@nut_sink
def ExportCAM(stuff, cfg, suffix="", nifty=False):
    for obj in stuff:
        (cams, filenames) = obj
        for i in range(len(filenames)):
            cam     = cams[0][i]
            cam_pos = cams[1][i]
            cam_neg = cams[2][i]

            if nifty:
                cam     = scale_cam(cam)
                cam_pos = scale_cam(cam_pos, 'automatic')
                cam_neg = scale_cam(-1.0*cam_neg, 'automatic')

                filename = filenames[i]

                if '-OD' in filename:
                    print('-OD: right eye')
                    lefteyeQ = False
                else:
                    print('-OS: left eye')
                    lefteyeQ = True

                export_nifty(cam,     cfg['OUT_DIR'] + '/' + filename + suffix + '_cam',     lefteyeQ)
                export_nifty(cam_pos, cfg['OUT_DIR'] + '/' + filename + suffix + '_cam_pos', lefteyeQ)
                export_nifty(cam_neg, cfg['OUT_DIR'] + '/' + filename + suffix + '_cam_neg', lefteyeQ)
            else:
                filename = filenames[i]
                np.save(cfg['OUT_DIR'] + '/' + filename + suffix + '_cam.npy', cam)
                np.save(cfg['OUT_DIR'] + '/' + filename + suffix + '_cam_pos.npy', cam_pos)
                np.save(cfg['OUT_DIR'] + '/' + filename + suffix + '_cam_neg.npy', cam_neg)
            
@nut_sink
def ConvertToNifty(stuff, cfg, suffix=""):
    for obj in stuff:
        (image, filename) = obj
        if '-OD' in filename:
            #print('-OD: right eye')
            lefteyeQ = False
        else:
            #print('-OS: left eye')
            lefteyeQ = True
        export_nifty(image, cfg['OUT_DIR'] + '/' + filename + suffix, lefteyeQ)


def scaleCAM(cam):
    cam2 = ni.rerange(cam, np.min(cam), np.max(cam), 0, 100, 'uint8')
    cam2 = resize_cube(cam2, [128, 64, 64])
    return cam2

def decompose_weights(w):

    def pos_rectify(x):
        if x>0.0:
            return x
        else:
            return 0.0*x

    def neg_rectify(x):
        if x<0.0:
            return x
        else:
            return 0.0*x

    w1 = np.apply_along_axis(pos_rectify, 0, w)
    w2 = np.apply_along_axis(neg_rectify, 0, w)

    return (w1, w2)

def convert_to_nifty(sample_splits, cfg, split_select):

    for c in map(code_split, split_select):

        filenames = sample_splits[c] >> Get(0) >> Collect()
            
        if cfg['ROI']=='MAC':
            suffix = '_MAC'
        elif cfg['ROI']=='ONH':
            suffix = '_ONH'
        else:
            print("error")

        (sample_splits[c] >>
         ReadVolumesOneBranch(cfg) >>
         Get(0)>>
         Zip(filenames) >>
         ConvertToNifty(cfg, suffix))

def generate_cam(samples, cfg, split_select, nifty=False):

    network = create_network(cfg)
    fold = cfg['FOLD_NO']
    network.load_weights(cfg['WEIGHTS_STEM'] + '_' + str(fold) + '.h5')
    w = get_weights(network, cfg)

    if cfg['TYPE']=='solo':
        read_cube = ReadVolumesOneBranch
    else:
        read_cube = ReadVolumes

    build_pred_batch = build_pred_batcher(cfg)

    if cfg['TYPE']=='solo':
        if cfg['ROI'] == 'MAC':
            suffix = "_MAC"
        elif cfg['ROI'] == 'ONH':
            suffix = "_ONH"
        else:
            print("error")
    elif cfg['TYPE'] == 'maconh_3heads':
        if cfg['ROI'] == 'MAC':
            suffix = '_maconh3MAC'
        elif cfg['ROI'] == 'ONH':
            suffix = '_maconh3ONH'
        elif cfg['ROI'] == 'MAConh':
            suffix = '_maconh3mergedMAC'
        elif cfg['ROI'] == 'macONH':
            suffix = '_maconh3mergedONH'
        else:
            print("error")

    for c in map(code_split, split_select):
        samples = sample_splits[c]

        def to_filename(x):
            return x[0] + '_' + str(fold) + '_' + str(c)

        filenames = samples >> Map(lambda x: to_filename(x)) >> Collect()
        filename_batches = list(chunks(filenames, cfg['BATCH_SIZE']))

        (samples >> 
         read_cube(cfg) >> 
         build_pred_batch >> 
         get_CAM_activations(network, cfg) >>
         CAM3D(w) >>
         Zip(filename_batches) >>
         ExportCAM(cfg, suffix, nifty))

def export_GAP_activations(sample_splits, cfg, split_select):

    network = create_network(cfg)
    fold = cfg['FOLD_NO']
    network.load_weights(cfg['WEIGHTS_STEM'] + '_' + str(fold) + '.h5')

    if cfg['TYPE']=='solo':
        read_cube = ReadVolumesOneBranch
    else:
        read_cube = ReadVolumes

    build_pred_batch = build_pred_batcher(cfg)

    if cfg['TYPE']=='solo':
        if cfg['ROI'] == 'MAC':
            suffix = "_MAC"
        elif cfg['ROI'] == 'ONH':
            suffix = "_ONH"
        else:
            print("error")
    elif cfg['TYPE'] == 'maconh_3heads':
        if cfg['ROI'] == 'MAC':
            suffix = '_maconh3MAC'
        elif cfg['ROI'] == 'ONH':
            suffix = '_maconh3ONH'
        elif cfg['ROI'] == 'MAConh':
            suffix = '_maconh3mergedMAC'
        elif cfg['ROI'] == 'macONH':
            suffix = '_maconh3mergedONH'
        elif cfg['ROI'] == 'MACONH':
            suffix = '_maconh3mergedMACONH'
        else:
            print("error")

    res = []
    for c in map(code_split, split_select):

        samples = sample_splits[c]
        filenames = samples  >> Get(0) >> Collect()
        filename_batches = list(chunks(filenames, cfg['BATCH_SIZE']))

        act_batches = (samples >>
                       read_cube(cfg) >>
                       build_pred_batch >>
                       get_GAP_activations(network, cfg) >>
                       Collect())

        zz_flatten = []
        for i in range(len(act_batches)):
            act_batch = act_batches[i].tolist()
            filename_batch = filename_batches[i]
            for j in range(len(act_batch)):
                row = [filename_batch[j], fold, c] + act_batch[j]
                zz_flatten = zz_flatten + [row]
        
        res = res + zz_flatten

        out_name = cfg['OUT_DIR'] + '/activations' + suffix + '.csv'
        with open(out_name, mode='w') as fp:
            csv_writer = csv.writer(fp, delimiter=',')
            csv_writer.writerows(res)

def export_weights(cfg):
    network = create_network(cfg)
    fold = cfg['FOLD_NO']
    network.load_weights(cfg['WEIGHTS_STEM'] + '_' + str(fold) + '.h5')
    w = get_weights(network, cfg, bias=True)

    if cfg['TYPE']=='solo':
        if cfg['ROI'] == 'MAC':
            suffix = "_MAC"
        elif cfg['ROI'] == 'ONH':
            suffix = "_ONH"
        else:
            print("error")
    elif cfg['TYPE'] == 'maconh_3heads':
        if cfg['ROI'] == 'MAC':
            suffix = '_maconh3MAC'
        elif cfg['ROI'] == 'ONH':
            suffix = '_maconh3ONH'
        elif cfg['ROI'] == 'MAConh':
            suffix = '_maconh3mergedMAC'
        elif cfg['ROI'] == 'macONH':
            suffix = '_maconh3mergedONH'
        elif cfg['ROI'] == 'MACONH':
            suffix = '_maconh3mergedMACONH'
        else:
            print("error")

    filename = cfg['OUT_DIR'] + '/' + 'weights' + suffix + '.csv'
    np.savetxt(filename, w,  delimiter=",")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', metavar='config_filename', help='name of config file')
    args = parser.parse_args()

    config_file = args.config_file
    cfg = nutsml.config.load_config(config_file)

    if cfg["DEVICE"] == "DLaaS":
        data_dir = os.environ["DATA_DIR"]
        out_dir  = os.environ["RESULT_DIR"]
        cfg['IMG_PATH'] = data_dir + '/'
        cfg['WEIGHTS_PATH'] = data_dir + '/'
        cfg['OUT_DIR'] = out_dir
    else:
        cfg['WEIGHTS_PATH'] = '/Users/hhyu/dev/cloud/projects-2019/maconh_bayes_analysis/'
        init_GPU(cfg) 

    fold = cfg['FOLD_NO']
    
    for job in cfg['JOBS']:
        print("#### job:", job)

        cfg['TYPE'] = job[1]

        train_samples, val_samples, test_samples = read_samples(fold, cfg)
        sample_splits = [train_samples, val_samples, test_samples]            

        if job[0] == 'convertOrigToNifty':
            # format ['convertOrigToNifty', 'solo', 'MAC/ONH', split_select]
            # split_select is a list that can have 'train', 'val', and 'test', or it can be 'all'
            cfg['ROI'] = job[2]
            
            if job[3] == 'all':
                split_select = ['train', 'val', 'test']
            else:
                split_select = [job[3]]

            convert_to_nifty(sample_splits, cfg, split_select)

        elif job[0] == 'export_weights':
            # format ['export_weights', 'solo/maconh_3heads', 'MAC/ONH/MACONH/MAConh/macONH', weights_directoy]
            cfg['ROI']  = job[2]
            cfg['WEIGHTS_STEM'] = cfg['WEIGHTS_PATH'] + job[3]
            export_weights(cfg)

        elif job[0] == 'export_activations':
            # format ['export_activations', 'solo/maconh_3heads', 'MAC/ONH/MACONH/MAConh/macONH', split_select, weights_directoy]
            cfg['ROI']  = job[2]

            if job[3] == 'all':
                split_select = ['train', 'val', 'test']

            cfg['WEIGHTS_STEM'] = cfg['WEIGHTS_PATH'] + '/' + job[4]

            export_GAP_activations(sample_splits, cfg, split_select)

        elif job[0] == 'generate_cams':
            # format ['generate_cams', 'solo/maconh_3heads', 'MAC/ONH/MAConh/macONH', split_select, weights_directory]
            cfg['ROI'] = job[2]

            if job[3] =='all':
                split_select = ['train', 'val', 'test']
            else:
                split_select = [job[3]]

            cfg['WEIGHTS_STEM'] = cfg['WEIGHTS_PATH'] + '/' + job[4]

            generate_cam(sample_splits, cfg, split_select, nifty=False)                

        elif job[0] == 'generate_cams_nifty':
            # format ['generate_cams_nifty', 'solo/maconh_3heads', 'MAC/ONH/MAConh/macONH', split_select, weights_directory]
            cfg['ROI'] = job[2]

            if job[3] =='all':
                split_select = ['train', 'val', 'test']
            else:
                split_select = [job[3]]

            cfg['WEIGHTS_STEM'] = cfg['WEIGHTS_PATH'] + job[4]
            print(cfg['WEIGHTS_PATH'])
            print(cfg['WEIGHTS_STEM'])

            generate_cam(sample_splits, cfg, split_select, nifty=True)
        else:
            print("error")
