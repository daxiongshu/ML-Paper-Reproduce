import tensorflow as tf
import copy
################################################################
#common flags
flags = tf.app.flags

flags.DEFINE_integer("start",None,"start point")
flags.DEFINE_integer("sample_size",None,"sample size")
flags.DEFINE_integer("num_units",None,"number of units")
#flags.DEFINE_integer("seq_len",None,"length of the sequence")
flags.DEFINE_integer("embedding_size",None,"embedding size")
flags.DEFINE_string('opt',None, 'optimizer')
flags.DEFINE_string('sol', None, 'id of the solution')
flags.DEFINE_string('cell', None, 'name of the rnn cells')
flags.DEFINE_integer("seed",0,"seed")
flags.DEFINE_integer("seeds",0,"seeds")
flags.DEFINE_string('run_name', None, 'name of the experiment')
flags.DEFINE_string('split_path', None, 'path of split file')
flags.DEFINE_string('embedding_path', None, 'Embedding path')
flags.DEFINE_integer("classes",None,"number of classes")
flags.DEFINE_integer("fold",None,"index of fold")
flags.DEFINE_integer("folds",None,"number of folds")
flags.DEFINE_string('log_path', None, 'Log path')
flags.DEFINE_integer('batch_size',None,"batch size")
flags.DEFINE_integer('threads',None,"number of threads")
flags.DEFINE_integer("augmentation",0,"data augmentation")
flags.DEFINE_float("learning_rate",None,"Learning rate")
flags.DEFINE_float("threshold",None,"threshold")
flags.DEFINE_float("keep_prob",None,"keep prob")
flags.DEFINE_string("metric",None,"metric")
flags.DEFINE_float("lambdax",None,"lambda for L2 regularization")
flags.DEFINE_float("epsilon",None,"epsilon in RL")
flags.DEFINE_integer("color",None,"Color channels")
flags.DEFINE_integer("epochs",None,"number of epochs")
flags.DEFINE_integer("pre_epochs",None,"pretrained number of epochs")
flags.DEFINE_string('paper', None, 'name of the paper')
flags.DEFINE_string('task', None, 'train or test')
flags.DEFINE_string('visualize',None,'visualize verbosity')
flags.DEFINE_string('save_path', None, 'path to save weights')
flags.DEFINE_integer('save_epochs',None,'for how many epochs are weights saved')
flags.DEFINE_string('load_path', None , 'path to load weights')
flags.DEFINE_string('net',None, 'net name')
flags.DEFINE_string('pred_path', None, 'name of prediction files')
flags.DEFINE_string('record_path', None, 'path of tf record')
flags.DEFINE_string('data_path', None, 'path of other data')
flags.DEFINE_string('input_path', None, 'input path')
flags.DEFINE_integer("width",None,"width of image to resize to")
flags.DEFINE_integer("height",None,"height of image to resize to")
flags.DEFINE_string('add_paths', None, 'additional input paths')
flags.DEFINE_string('add_record_paths', None, 'additional records')
flags.DEFINE_float("momentum",None,"momentum")
#####################################################################

#####################################################################

#####################################################################

FLAGS = flags.FLAGS

