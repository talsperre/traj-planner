import argparse
parser = argparse.ArgumentParser()

################ Model Options ################################
parser.add_argument('-init_type', help='Weight initialization for the linear layers', type=str.lower,
                    choices=['xavier', 'default'], default='default')
parser.add_argument('-activation', help='Activation function to be used', type=str.lower,
                    choices=['relu', 'selu'], default='relu')
parser.add_argument('-num_channels', help='Number of channels', type=int, default=2)
parser.add_argument('-image_width', help='Width of the input image', type=int, default=360)
parser.add_argument('-image_height', help='Height of the input image', type=int, default=240)

################### Hyperparameters ###########################
parser.add_argument('-lr', help='Learning rate', type=float, default=1e-4)
parser.add_argument('-batch_size', help='Batch Size', type=int, default=32)
parser.add_argument('-momentum', help='Momentum', type=float, default=0.9)
parser.add_argument('-weight_decay', help='Weight decay', type=float, default=0.)
parser.add_argument('-lr_decay', help='Learning rate decay factor', type=float, default=0.)
parser.add_argument('-num_epochs', help='Number of epochs',
                    type=int, default=1)
parser.add_argument('-beta1', help='beta1 for ADAM optimizer', type=float, default=0.9)
parser.add_argument('-beta2', help='beta2 for ADAM optimizer', type=float, default=0.999)
parser.add_argument('-opt_method', help='Optimization method : adam | sgd | amsgrad',
                    type=str.lower, choices=['adam', 'sgd', 'amsgrad'], default='adam')
parser.add_argument('-batchnorm', help='Use batchnorm', default=False)

################### Dataset ######################################
parser.add_argument('-train_path', help='Train dataset path')
parser.add_argument('-test_path', help='Test dataset path')
parser.add_argument('-time_hist', help='No of timesteps of past trajectory', type=int, default=20)
parser.add_argument('-time_fut', help='No of timesteps of future trajectory', type=int, default=40)
parser.add_argument('-sampling_rate', help='Sampleing rate', type=int, default=2)

arguments = parser.parse_args()
