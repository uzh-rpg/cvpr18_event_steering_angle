import gflags
import numpy as np
import os
import sys
import json
from unipath import Path

from keras import backend as K
import tensorflow as tf

import utils
from constants import TEST_PHASE
from common_flags import FLAGS


# Functions to evaluate steering prediction

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions,
                                                 real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance


def compute_sq_residuals(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis = -1)
    print("MSE = {}".format(sr))
    return sq_res


def compute_rmse(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values), axis=0)
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse


def compute_highest_regression_errors(predictions, real_values):
    """
    Compute the indexes with highest error
    """
    n_errors = 5
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.sqrt(np.square(predictions - real_values))
    highest_errors = np.sort(sq_res, axis=None)[-n_errors:]
    print("=============")
    print("Highest errors")
    print(highest_errors)
    print("=============")
    return highest_errors


def random_regression_baseline(real_values):
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)


def evaluate_regression(predictions, real_values):
    results = {}
    results['evas'] = compute_explained_variance(predictions, real_values)
    results['rmse'] = compute_rmse(predictions, real_values).tolist()
    results['h_error'] = compute_highest_regression_errors(predictions, real_values).tolist()
    return results


def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)
    
    seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Generate testing data
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

    ## Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))-1

    predictions, ground_truth = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)


    print('----------------------------------')
    print('Prediction std is {}'.format(np.std(predictions)))
    print('----------------------------------')

    # Transformed predictions (network output)
    u_dict = {'trasformed_predicted': predictions,
              'transfomed_constant': np.ones_like(ground_truth) * np.mean(ground_truth)}

    # Evaluate transformed predictions (won't be saved)
    results_dict = {}
    for name, pred in u_dict.items():
        print("------------------------")
        print("Evaluating {}".format(name))
        evaluation = evaluate_regression(pred, ground_truth)
        print("------------------------")
        results_dict[name] = [evaluation]

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
    pred_std = (predictions - min_bound)/(max_bound - min_bound)
    pred_steer = pred_std*(maxs - mins) + mins
    #pred_steer = np.expand_dims(pred_steer, axis = -1)

    # Undo transformation for ground-truth (only for steering)
    gt_std = (ground_truth - min_bound)/(max_bound - min_bound)
    steer_gt = gt_std*(maxs - mins) + mins
    #steer_gt = np.expand_dims(gt_steer, axis=-1)

    # Compute random and constant baselines for steerings
    random_steerings = random_regression_baseline(steer_gt).ravel()
    constant_steerings = constant_baseline(steer_gt).ravel()

    # Create dictionary of baselines
    baseline_dict = {'predicted': pred_steer,
                     'random': random_steerings,
                     'constant': constant_steerings}

    # Evaluate detransformed predictions: EVA, residuals, and highest errors
    results_dict = {}
    for name, pred in baseline_dict.items():
        print("------------------------")
        print("Evaluating {}".format(name))
        evaluation = evaluate_regression(pred, steer_gt)
        print("------------------------")
        results_dict[name] = [evaluation]

    utils.write_to_file(results_dict, os.path.join(FLAGS.experiment_rootdir, 'test_results.json'))

    # Write predicted and real steerings
    steer_dict = {'pred_steerings': pred_steer.tolist(),
                  'real_steerings': steer_gt.tolist()}
    utils.write_to_file(steer_dict, os.path.join(FLAGS.experiment_rootdir,
                                               'predicted_and_real_steerings.json'))


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
