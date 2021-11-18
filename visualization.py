import numpy as np
import cv2
import gflags
import os
from vis.visualization import visualize_cam, overlay
from keras.utils.generic_utils import Progbar
import sys
import tensorflow as tf
from unipath import Path

from common_flags import FLAGS
import img_utils
import utils
import re

from viewer import process_dvs_as_rb

gflags.DEFINE_string("input_imgs_dir", "","Input images directory")
gflags.DEFINE_string("output_dir", "", "Directory where to write images")


modifiers = [None]


def recursive_list(subpath):
    return sorted(os.walk(subpath), key=lambda tpl: tpl[0])


def load_fnames(dir_subpath, frame_mode):
    # Steering angle is not predicted for the first APS DIFF frame
    steerings_filename = os.path.join(dir_subpath, "sync_steering.txt")
    try:
        outputs = np.loadtxt(steerings_filename, delimiter=',',
                                   skiprows=1)
    except:
        raise IOError("GT files not found")

    filenames = []

    if frame_mode == 'aps_diff':
        outputs = outputs[1:]

    # Now fetch all images in the image subdir
    if frame_mode == 'dvs':
        image_dir_path = os.path.join(dir_subpath, "dvs")
    elif frame_mode == 'aps':
        image_dir_path = os.path.join(dir_subpath, "aps")
    else:
        image_dir_path = os.path.join(dir_subpath, "aps_diff")

    for root, _, files in recursive_list(image_dir_path):
        sorted_files = sorted(files,
                key = lambda fname: int(re.search(r'\d+',fname).group()))
        for frame_number, fname in enumerate(sorted_files):
           is_valid = False
           for extension in {'png'}:
               if fname.lower().endswith('.' + extension):
                   if np.abs(outputs[frame_number][3]) < 2.30e1:
                       break
                   else:
                       is_valid = True
                       break

           if is_valid:
               absolute_path = os.path.join(root, fname)
               filenames.append(absolute_path)

    print("Found {} filenames to analyze".format(len(filenames)))
    assert len(filenames) > 0, "No filenames found"
    return filenames


def visualize_dvs_img(fname, target_size=None, crop_size=None):
    img = cv2.imread(fname)

    if crop_size:
        img = img_utils.image_crop(img, crop_size[0], crop_size[1])

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, target_size)

    img = process_dvs_as_rb(img, constant=30)

    return img


def read_percentiles(frame_mode):
    if frame_mode == 'dvs':
        # Load percentiles for positive and negative event normalization
        try:
            percentiles = np.loadtxt(os.path.join(Path(FLAGS.train_dir).parent,
                                                  'percentiles.txt'), usecols=0,
                                   skiprows=1)
        except:
            raise IOError("Percentiles file not found")
    else:
        percentiles = None

    return percentiles

def _main():

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir,
                                   FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Check that output dir actually exists
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir,
                                     FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")

    target_size = (FLAGS.img_width, FLAGS.img_height)
    crop_size = (FLAGS.crop_img_width, FLAGS.crop_img_height)
    frame_mode = FLAGS.frame_mode

    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(FLAGS.batch_size, trainable=False, name='k_mse', dtype=tf.int32)

    model.compile(loss=[utils.hard_mining_mse(model.k_mse)],
                      optimizer='adam', decay=1e-4, lr=FLAGS.initial_lr,
                      metrics=[utils.steering_loss, utils.pred_std])

    filenames = load_fnames(FLAGS.input_imgs_dir, frame_mode)
    percentiles = read_percentiles(frame_mode)


    progbar = Progbar(target=len(filenames))
    for n, fname in enumerate(filenames):

        img = img_utils.load_img(fname, frame_mode, percentiles,
                                 target_size, crop_size)
        if frame_mode == 'dvs':
            colored = visualize_dvs_img(fname, target_size, crop_size)
        else:
            colored = cv2.imread(fname, 3)

        if frame_mode == 'aps':
            img = np.asarray(img / 255.0, dtype = np.float32)
        for i, modifier in enumerate(modifiers):
            heatmap = visualize_cam(model, layer_idx=-1, filter_indices=0,
                                    seed_input=img, grad_modifier=modifier)
            # Overlay is used to alpha blend heatmap onto img
            result_fname = os.path.join(FLAGS.output_dir,
                                        os.path.basename(fname))
            new_img = cv2.cvtColor(overlay(colored, heatmap, alpha=0.6),
                                   cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_fname, new_img)

        progbar.update(n)

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
