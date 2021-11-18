"""
Processes a new video sequence to predict the steering angle for each frame.
DroneDataGenerator is used to generate data from the new sequence, so that
'video_dir' must contain a single experiment with the same structure as the
training, validation and testing data:
name_of_experiment/
    exp_1/
        dvs/
        aps/
        aps_diff/
        sync_steering
        
If the sequence does not have groundtruth, you must create it because DroneDataGenerator
expects a txt file. For simplicity, just create sync_steering.txt with as many zeros
as images in the sequence.
"""


import gflags
import numpy as np
import os
import sys
from unipath import Path
import json

from keras import backend as K

import utils
from constants import TEST_PHASE
from common_flags import FLAGS


def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Generate data
    if FLAGS.frame_mode == 'dvs' or FLAGS.frame_mode == 'aps_diff':
        test_datagen = utils.DroneDataGenerator()
    else:
        test_datagen = utils.DroneDataGenerator(rescale = 1./255)
        
    test_generator = test_datagen.flow_from_directory(FLAGS.test_dir,
                          shuffle=False,
                          frame_mode = FLAGS.frame_mode,
                          target_size=(FLAGS.img_height, FLAGS.img_width),
                          crop_size=(FLAGS.crop_img_height, FLAGS.crop_img_width),
                          batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except IOError as e:
       print("Impossible to find weight path. Returning untrained model")


    # Compile model
    model.compile(loss='mse', optimizer='sgd')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))

    predictions, ground_truth = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)
    
    
    # Steering boundaries seen in data
    json_dict_fname = os.path.join(
        Path(os.path.realpath(FLAGS.test_dir)).parent,
        'scalers_dict.json')

    with open(json_dict_fname, 'r') as f:
        scalers_dict = json.load(f)

    mins = np.array(scalers_dict['mins'])
    maxs = np.array(scalers_dict['maxs'])

    # Range of the transformed data
    min_bound = -1.0
    max_bound = 1.0

    # Undo transformation for predicitons (only for steering)
    pred_std = (predictions[:,0] - min_bound)/(max_bound - min_bound)
    pred_steer = pred_std*(maxs[0] - mins[0]) + mins[0]
    pred_steer = np.expand_dims(pred_steer, axis = -1)
    
    # Undo transformation for ground-truth (only for steering)
    gt_std = (ground_truth[:,0] - min_bound)/(max_bound - min_bound)
    gt_steer = gt_std*(maxs[0] - mins[0]) + mins[0]
    steer_gt = np.expand_dims(gt_steer, axis=-1)


    # Write predicted and real steerings
    steer_dict = {'pred_steerings': pred_steer.tolist(),
                 'real_steerings': steer_gt.tolist()}
    utils.write_to_file(steer_dict, os.path.join(FLAGS.test_dir,
                                               'predicted_and_real_steerings_' + FLAGS.frame_mode + '.json'))



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
