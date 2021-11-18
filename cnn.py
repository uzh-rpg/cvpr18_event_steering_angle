import tensorflow as tf
import numpy as np
import os
import sys
import gflags

from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras

import logz
import cnn_models
import utils
import log_utils
from common_flags import FLAGS
from constants import TRAIN_PHASE



def getModel(img_width, img_height, img_channels, output_dim, weights_path):
    """
    Initialize model.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       weights_path: Path to pre-trained model.

    # Returns
       model: A Model instance.
    """
    if FLAGS.imagenet_init:
        model = cnn_models.resnet50(img_width,
                                    img_height, img_channels, output_dim)
    else:
        model = cnn_models.resnet50_random_init(img_width,
                                    img_height, img_channels, output_dim)


    if weights_path:
        #try:
        model.load_weights(weights_path)
        print("Loaded model from {}".format(weights_path))
        #except:
        #    print("Impossible to find weight path. Returning untrained model")

    return model


def trainModel(train_data_generator, val_data_generator, model, initial_epoch):
    """
    Model training.

    # Arguments
       train_data_generator: Training data generated batch by batch.
       val_data_generator: Validation data generated batch by batch.
       model: Target image channels.
       initial_epoch: Dimension of model output.
    """

    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(FLAGS.batch_size, trainable=False, name='k_mse', dtype=tf.int32)

    # Configure training process
    optimizer = keras.optimizers.Adam(lr=FLAGS.initial_lr, decay=1e-4)
    model.compile(loss=[utils.hard_mining_mse(model.k_mse)], optimizer=optimizer,
                  metrics=[utils.steering_loss, utils.pred_std])

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_rootdir, 'weights_{epoch:03d}.h5')
    writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_steering_loss',
                                     save_best_only=True, save_weights_only=True)

    # Save model every 'log_rate' epochs.
    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    saveModelAndLoss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir,
                                            period=FLAGS.log_rate,
                                            batch_size=FLAGS.batch_size,
                                            factor=FLAGS.lr_scale_factor)

    # Train model
    steps_per_epoch = np.minimum(int(np.ceil(
        train_data_generator.samples / FLAGS.batch_size)), 2000)
    validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))-1

    model.fit_generator(train_data_generator,
                        epochs=FLAGS.epochs, steps_per_epoch = steps_per_epoch,
                        callbacks=[writeBestModel, saveModelAndLoss],
                        validation_data=val_data_generator,
                        validation_steps = validation_steps,
                        initial_epoch=initial_epoch)


def _main():

    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0,2*31-1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)

    K.set_learning_phase(TRAIN_PHASE)

    # Create the experiment rootdir if not already there
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    # Cropped image dimensions
    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height

    # Output dimension (one for steering)
    output_dim = 1

    # Input image channels
    # - DVS frames: 2 channels (first one for positive even, second one for negative events)
    # - APS frames: 1 channel (grayscale images)
    # - APS DIFF frames: 1 channel (log(I_1) -  log(I_0))
    if FLAGS.frame_mode == 'dvs':
        img_channels = 3
    else:
        img_channels = 3


    # Generate training data with real-time augmentation
    if FLAGS.frame_mode == 'dvs':
        train_datagen = utils.DroneDataGenerator()
    elif FLAGS.frame_mode == 'aps':
        train_datagen = utils.DroneDataGenerator(rotation_range = 0.2,
                                                 rescale = 1./255,
                                                 width_shift_range = 0.2,
                                                 height_shift_range=0.2)
    else:
        train_datagen = utils.DroneDataGenerator(rotation_range = 0.2,
                                                 width_shift_range = 0.2,
                                                 height_shift_range=0.2)

    train_generator = train_datagen.flow_from_directory(FLAGS.train_dir,
                                                        is_training=True,
                                                        shuffle = True,
                                                        frame_mode = FLAGS.frame_mode,
                                                        target_size=(img_height, img_width),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size = FLAGS.batch_size)

    # Generate validation data with real-time augmentation
    if FLAGS.frame_mode == 'dvs' or FLAGS.frame_mode == 'aps_diff':
        val_datagen = utils.DroneDataGenerator()
    else:
        val_datagen = utils.DroneDataGenerator(rescale = 1./255)

    val_generator = val_datagen.flow_from_directory(FLAGS.val_dir,
                                                        shuffle = False,
                                                        frame_mode = FLAGS.frame_mode,
                                                        target_size=(img_height, img_width),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size = FLAGS.batch_size)
    # output dim
    assert train_generator.output_dim == val_generator.output_dim, \
                        " Not macthing output dimensions."
    output_dim = train_generator.output_dim

    # Weights to restore
    weights_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    initial_epoch = 0
    if not FLAGS.restore_model:
        # In this case weights will start from random
        weights_path = None
    else:
        # In this case weigths will start from the specified model
        initial_epoch = FLAGS.initial_epoch

    # Define model
    model = getModel(img_width, img_height, img_channels,
                        output_dim, weights_path)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.modelToJson(model, json_model_path)

    # Train model
    trainModel(train_generator, val_generator, model, initial_epoch)


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
