import gflags



FLAGS = gflags.FLAGS

# Random seed
gflags.DEFINE_bool('random_seed', True, 'Random seed')

# Input
gflags.DEFINE_integer('img_width', 200, 'Target Image Width')
gflags.DEFINE_integer('img_height', 200, 'Target Image Height')

gflags.DEFINE_integer('crop_img_width', 200, 'Cropped image widht')
gflags.DEFINE_integer('crop_img_height', 200, 'Cropped image height')

gflags.DEFINE_string('frame_mode', "dvs", 'Load mode for images, either '
                     'dvs, aps or aps_diff')
gflags.DEFINE_string('visual_mode', "red_blue", 'Mode for video visualization, either '
                     'red_blue or grayscale')

# Training
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 100, 'Number of epochs for training')
gflags.DEFINE_integer('log_rate', 10, 'Logging rate for full model (epochs)')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_float('initial_lr', 1e-4, 'Initial learning rate for adam')
gflags.DEFINE_float('lr_scale_factor', 0.5, 'Reducer factor for learning rate '
                    'when loss stagnates.')
gflags.DEFINE_bool('imagenet_init', True, 'Initialization from imagenet weights')

# Files
gflags.DEFINE_string('experiment_rootdir', "./model", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "../training", 'Folder containing'
                     ' training experiments')
gflags.DEFINE_string('val_dir', "../validation", 'Folder containing'
                     ' validation experiments')
gflags.DEFINE_string('test_dir', "../testing", 'Folder containing'
                     ' testing experiments')
gflags.DEFINE_string('video_dir', "../video_1", 'Folder containing'
                     ' only one experiment to be processed')
gflags.DEFINE_string('exp_name', "exp_1", 'Name of the experiment'
                     ' to be processed')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_fname', "model_weights.h5", '(Relative) '
                                          'filename of model weights')
gflags.DEFINE_string('json_model_fname', "model_struct.json",
                          'Model struct json serialization, filename')

