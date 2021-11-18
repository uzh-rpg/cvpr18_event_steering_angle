# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:49:08 2017
@author: ana
"""

'''
Results video generator Udacity Challenge 2
Original By: Comma.ai Revd: Chris Gundling
'''

import cv2
import glob
import sys
import os
import numpy as np
import json
import gflags
import re

from common_flags import FLAGS


def process_dvs_as_grayscale(img, climit=[-100,100]):
    pos_img = (10*img[:,:,0]).astype('float32')
    neg_img = (10*img[:,:,-1]).astype('float32')
    gray_img = pos_img - neg_img
    gray_img = (np.clip(gray_img, climit[0], climit[1]).astype('float32')+127).astype('uint8')
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    return gray_img


def process_dvs_as_rb(img, constant, climit=[0,255]):
    img[:,:,0] = constant*img[:,:,0]
    img[:,:,-1] = constant*img[:,:,-1]
    img = np.clip(img, climit[0], climit[1]).astype('uint8')
    return img



def get_data(exp_dir, img_height, img_width, img_channels, frame_mode, visual_mode):

    # Read images
    img_files = [os.path.basename(x) for x in glob.glob(exp_dir + "/" + frame_mode + "/*")]
    test_x = np.zeros((len(img_files),img_height, img_width, img_channels))
    sorted_files = sorted(img_files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
    for i,fname in enumerate(sorted_files):
        img = cv2.imread(os.path.join(exp_dir, frame_mode, fname))
        if frame_mode=='dvs':
            if visual_mode == 'grayscale':
                img = process_dvs_as_grayscale(img)
            else:
                img = process_dvs_as_rb(img)

        elif frame_mode=='aps':
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        else:
            input_img = (np.log(img[:,:,-1] + 1e-3) - np.log(img[:,:,0] + 1e-3))
            img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)

        test_x[i] = img
    return test_x


def plot_steering(img, pred_steer, real_steer):
    c, r = (173, 130), 65 #center, radius

    # Draw circle
    cv2.circle(img, c, r, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (c[0]-r+5, c[1]), (c[0]-r, c[1]), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (c[0]+r-5, c[1]), (c[0]+r, c[1]), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (c[0], c[1]-r+5), (c[0], c[1]-r), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (c[0], c[1]+r-5), (c[0], c[1]+r), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Draw real steering
    real_rad = + real_steer / 180. * np.pi + np.pi / 2
    t = (c[0] + int(np.cos(real_rad) * r), c[1] - int(np.sin(real_rad) * r))
    cv2.line(img, c, t, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(img, 'GT', (c[0]-r-60, c[1]-r-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,lineType=cv2.LINE_AA)
    cv2.putText(img, '%0.1f deg' % real_steer, (c[0]-r-60, c[1]-r-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,lineType=cv2.LINE_AA)

    # Draw predicted steering
    pred_rad = + pred_steer / 180. * np.pi + np.pi / 2
    t = (c[0] + int(np.cos(pred_rad) * r), c[1] - int(np.sin(pred_rad) * r))
    cv2.line(img, c, t, (0,255,0), 2, lineType=cv2.LINE_AA)

    if FLAGS.frame_mode == 'dvs':
        cv2.putText(img, 'DVS', (c[0]+35, c[1]-r-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,lineType=cv2.LINE_AA)
    elif FLAGS.frame_mode =='aps':
        cv2.putText(img, 'APS', (c[0]+35, c[1]-r-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,lineType=cv2.LINE_AA)
    else:
        cv2.putText(img, 'APS_DIFF', (c[0]+35, c[1]-r-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,lineType=cv2.LINE_AA)

    cv2.putText(img, '%0.1f deg' % pred_steer, (c[0]+35, c[1]-r-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,lineType=cv2.LINE_AA)

#    if FLAGS.frame_mode == 'dvs':
#        # Draw predicted steering with DVS
#        dvs_pred_rad = + dvs_steer / 180. * np.pi + np.pi / 2
#        t = (c[0] + int(np.cos(dvs_pred_rad) * r), c[1] - int(np.sin(dvs_pred_rad) * r))
#        cv2.line(img, c, t, (0,255,0), 2, lineType=cv2.LINE_AA)
#        cv2.putText(img, 'DVS', (c[0]+35, c[1]-r-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,lineType=cv2.LINE_AA)
#        cv2.putText(img, '%0.1f deg' % dvs_steer, (c[0]+35, c[1]-r-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,lineType=cv2.LINE_AA)
#
#    elif FLAGS.frame_mode =='aps':
#        # Draw predicted steering with APS
#        aps_pred_rad = + 0 / 180. * np.pi + np.pi / 2
#        t = (c[0] + int(np.cos(aps_pred_rad) * r), c[1] - int(np.sin(aps_pred_rad) * r))
#        cv2.line(img, c, t, (0,255,0), 2, lineType=cv2.LINE_AA)
#        cv2.putText(img, 'APS', (c[0]+35, c[1]-r-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,lineType=cv2.LINE_AA)
#        cv2.putText(img, '%0.1f deg' % 0, (c[0]+35, c[1]-r-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,lineType=cv2.LINE_AA)
#
#    else:
#        # Draw predicted steering with DVS
#        dvs_pred_rad = + dvs_steer / 180. * np.pi + np.pi / 2
#        t = (c[0] + int(np.cos(dvs_pred_rad) * r), c[1] - int(np.sin(dvs_pred_rad) * r))
#        cv2.line(img, c, t, (0,0,255), 2, lineType=cv2.LINE_AA)
#        cv2.putText(img, 'DVS', (c[0]-30, c[1]-r-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1,lineType=cv2.LINE_AA)
#        cv2.putText(img, '%0.1f deg' % dvs_steer, (c[0]-30, c[1]-r-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1,lineType=cv2.LINE_AA)
#
#        # Draw predicted steering with APS
#        aps_pred_rad = + 0 / 180. * np.pi + np.pi / 2
#        t = (c[0] + int(np.cos(aps_pred_rad) * r), c[1] - int(np.sin(aps_pred_rad) * r))
#        cv2.line(img, c, t, (0,0,255), 2, lineType=cv2.LINE_AA)
#        cv2.putText(img, 'APS', (c[0]+r, c[1]-r-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1,lineType=cv2.LINE_AA)
#        cv2.putText(img, '%0.1f deg' % 0, (c[0]+r, c[1]-r-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1,lineType=cv2.LINE_AA)

    return img



def _main():

    # Path to images
    exp_dir = os.path.join(FLAGS.test_dir, 'exp_1')

    # Read ground truth
    steerings_filename = os.path.join(exp_dir, "sync_steering.txt")
    try:
        gt = np.loadtxt(steerings_filename, delimiter=',', skiprows=1)
    except:
        raise IOError("Steering file not found")


    # Prepare steering data
    fname_steer = os.path.join(FLAGS.test_dir, 'predicted_and_real_steerings.json')
    with open(fname_steer,'r') as f:
        dict_steerings = json.load(f)
    pred_steerings = np.array(dict_steerings['pred_steerings'])
    real_steerings = np.array(dict_steerings['real_steerings'])
    n_predictions = pred_steerings.shape[0]


    # Prepare images
    img_height, img_width, img_channels = 260, 346, 3

    # Always visualize APS frames
    aps_images = get_data(exp_dir, img_height, img_width, img_channels, 'aps',
                              FLAGS.visual_mode)
    aps_images = aps_images[-n_predictions:,:,:,:]
    print('APS data shape:', aps_images.shape)

    if FLAGS.frame_mode == 'dvs':
        # Prepare DVS images
        dvs_images = get_data(exp_dir, img_height, img_width, img_channels, FLAGS.frame_mode,
                              FLAGS.visual_mode)
        dvs_images = dvs_images[-n_predictions:,:,:,:]
        num_images = dvs_images.shape[0]
        print('DVS data shape:', dvs_images.shape)

    elif FLAGS.frame_mode == 'aps_diff':
        # Prepare APS images
        aps_diff_images = get_data(exp_dir, img_height, img_width, img_channels, FLAGS.frame_mode,
                              FLAGS.visual_mode)
        aps_diff_images = aps_diff_images[-n_predictions:,:,:,:]
        num_images = aps_diff_images.shape[0]
        print('APS data shape:', aps_diff_images.shape)


    # Run through all images
    for i in range(num_images):

        # Check if velocity is 0
        if np.abs(gt[i][3]) >= 2.30e1:
            pred_steer = float(pred_steerings[i])
            real_steer = float(real_steerings[i])
        else:
            if i==0:
                pred_steer = 0
                real_steer = 0
            else:
                pred_steer = float(pred_steerings[i-1])
                real_steer = float(real_steerings[i-1])


        # Show DVS and APS jointly
        if FLAGS.frame_mode == 'dvs':
            dvs = dvs_images[i]
            aps = aps_images[i]
            aps_steer = plot_steering(aps, pred_steer, real_steer)
            output_img = np.concatenate((aps_steer, dvs), axis=1)
            output_path = os.path.join(FLAGS.test_dir, "dvs_video")

        # Show APS only
        elif FLAGS.frame_mode == 'aps':
            # Draw predicted steering in APS frame
            aps = aps_images[i]
            output_img = plot_steering(aps, pred_steer, real_steer)
            output_path = os.path.join(FLAGS.test_dir, "aps_video")

        # Show APS_DIFF and APS jointly
        else:
            aps_diff = aps_diff_images[i]
            aps = aps_images[i]
            aps_steer = plot_steering(aps, pred_steer, real_steer)
            output_img = np.concatenate((aps_steer, aps_diff), axis=1)
            output_path = os.path.join(FLAGS.test_dir, "aps_diff_video")

        # Save frame as png
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img_name = "frame_" + str(i).zfill(5)  + ".png"
        cv2.imwrite(os.path.join(output_path, img_name),output_img)


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
