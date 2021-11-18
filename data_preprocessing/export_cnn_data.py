'''
Export DVS frames, APS frames, APS diff frames (difference of grayscale frames), and steering angles to be used by
the networks.
'''

import h5py
import numpy as np
import cv2
import os
import argparse
import glob
from itertools import groupby
from operator import itemgetter


def split_sequence(data):
    sequences = []
    for k, g in groupby(enumerate(data), lambda (i, x): i-x):
        sequences.append(map(itemgetter(1), g))
    return sequences
    
        
def export_data(f_in, idxs, out_path, pos_inf, pos_sup, neg_inf, neg_sup):
    
    # Non-image data
    data = np.zeros((len(idxs), 4))
    
    for key in f_in.keys():
        key = str(key)
        
        # Export DVS frames
        if key == 'dvs_frame':
            dvs_path = os.path.join(out_path, 'dvs')
            os.makedirs(dvs_path)

            images = f_in[key].value[idxs]
            for i in range(images.shape[0]):
                new_img = np.zeros((images.shape[1], images.shape[2], 3), dtype=np.uint8)
                event_img = images[i]
                
                # Positive events to channel 0
                pos_img = event_img[:,:,0]
                index_p = pos_img > 0
                pos_img[index_p] = np.clip(pos_img[index_p], pos_inf, pos_sup)
                new_img[:,:,0] = pos_img
                
                # Negative events to channel 1
                neg_img = event_img[:,:,1]
                index_n = neg_img > 0
                neg_img[index_n] = np.clip(neg_img[index_n], neg_inf, neg_sup)
                new_img[:,:,-1] = neg_img
                
                # Save DVS frame
                img_name = "frame_" + str(i).zfill(5)  + ".png"
                cv2.imwrite(os.path.join(dvs_path, img_name),new_img)
        
        # Export APS frames and APS diff frames (difference of grayscale frames)
        elif key == 'aps_frame':
            aps_path = os.path.join(out_path, 'aps')
            os.makedirs(aps_path)

            aps_diff_path = os.path.join(out_path, 'aps_diff')
            os.makedirs(aps_diff_path)

            images = f_in[key].value[idxs]
            images = np.asarray(images, dtype = np.uint8)
            for i in range(images.shape[0]):

                # Save APS frames
                img_name = "frame_" + str(i).zfill(5)  + ".png"
                cv2.imwrite(os.path.join(aps_path, img_name),images[i,:,:])
                
                # Save APS diff frames
                if i > 0:
                    new_img = np.zeros((images.shape[1], images.shape[2], 3), dtype=np.uint8)
                    new_img[:,:,0] = images[i-1,:,:]
                    new_img[:,:,-1] = images[i,:,:]
                    cv2.imwrite(os.path.join(aps_diff_path, img_name),new_img) 
                    
        
        # Steering, torque, engine speed, vehicle speed associated to DVS and APS frames
        elif key == 'steering_wheel_angle':
            steer = f_in[key].value[idxs]
            data[:,0] = steer
        elif key == 'torque_at_transmission':
            torque = f_in[key].value[idxs]
            data[:,1] = torque
        elif key == 'engine_speed':
            eng_speed = f_in[key].value[idxs]
            data[:,2] = eng_speed
        elif key == 'vehicle_speed':
            veh_speed = f_in[key].value[idxs]
            data[:,3] = veh_speed   
                
    # Save steering angles
    txt_name = os.path.join(out_path, 'sync_steering.txt')
    np.savetxt(txt_name, data, delimiter=',', header='steering, torque, engine_velocity, vehicle_velocity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', help='Path to frame-based hdf5 files.')
    args = parser.parse_args()
    
    # Load percentiles
    try:
        percentiles = np.loadtxt(os.path.join(args.source_folder, 'percentiles.txt'), usecols=0, skiprows=1)
    except:
        raise IOError("Percentiles file not found")
    pos_inf = percentiles[1] # Inferior percentile for positive events
    pos_sup = percentiles[2] # Superior percentile for positive events
    neg_inf = percentiles[4] # Inferior percentile for negative events
    neg_sup = percentiles[5] # Superior percentile for negative events
           
    # For every recording/hdf5 file
    recordings = glob.glob(args.source_folder + '/*.hdf5')
    for rec in recordings:
        f_in = h5py.File(rec, 'r')
        
        # Name of the experiment
        exp_name = rec.split('.')[-2]
        exp_name = exp_name.split('/')[-1]
        
        # Get training sequences
        train_idxs = np.ndarray.tolist(f_in['train_idxs'].value)
        train_sequences = split_sequence(train_idxs)
        for i, train_seq in enumerate(train_sequences):
            output_path = os.path.join(args.source_folder, 'training', exp_name + str(i))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            export_data(f_in, train_seq, output_path, pos_inf, pos_sup, neg_inf, neg_sup)

        # Get testing sequences
        test_idxs = np.ndarray.tolist(f_in['test_idxs'].value)
        test_sequences = split_sequence(test_idxs)
        for j, test_seq in enumerate(test_sequences):
            output_path = os.path.join(args.source_folder, 'testing', exp_name + str(j))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            export_data(f_in, test_seq, output_path, pos_inf, pos_sup, neg_inf, neg_sup)
        
        f_in.close()

