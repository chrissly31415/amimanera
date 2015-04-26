import theano
import numpy as np

from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum
from lasagne.objectives import Objective
from lasagne.regularization import l2
#from lasagne.regularization import l1
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

import cPickle as pickle

def plotNN(net1):
    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    plt.plot(train_loss, linewidth=3, label="train")
    plt.plot(valid_loss, linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(1e-3, 1e-2)
    plt.yscale("log")
    plt.show()
 
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        print 'NEW VALUE:',new_value
        getattr(nn, self.name).set_value(new_value)

def float32(k):
    return np.cast['float32'](k)


Maxout = layers.pool.FeaturePoolLayer

class L2Regularization(Objective):
  
    def __init__(self, input_layer, loss_function=None, aggregation='mean',**args):
	Objective.__init__(self, input_layer, loss_function, aggregation)
	self.alpha=args['alpha']
    
    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):
        loss = super(L2Regularization, self).get_loss(input=input,target=target, deterministic=deterministic, **kwargs)
        if not deterministic:
            return loss + self.alpha * l2(self.input_layer)
        else:
            return loss

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()
	  
	  
nnet2 = NeuralNet(

    layers=[ 
      ('input', layers.InputLayer),      
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer),
	],
    # layer parameters:
    input_shape=(None,92),  # 96x96 input pixels per batch

    hidden1_num_units=500,  # number of units in hidden layer
    hidden1_nonlinearity=nonlinearities.rectify,
    dropout1_p=0.5,

    hidden2_num_units=500,
    hidden2_nonlinearity=nonlinearities.rectify,
    dropout2_p=0.5,

    hidden3_num_units=500,
    hidden3_nonlinearity=nonlinearities.rectify,
    dropout3_p=0.5,

    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    objective=L2Regularization,
    objective_alpha=0.000005,

    eval_size=0.0,
    batch_iterator_train=BatchIterator(batch_size=1024),
    batch_iterator_test=BatchIterator(batch_size=1024),

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
	AdjustVariable('update_learning_rate', start=0.03, stop=0.01),
	AdjustVariable('update_momentum', start=0.9, stop=0.999),
	#EarlyStopping(patience=200),
	],


    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    verbose=1,
    )

nnet3 = NeuralNet(

    layers=[ 
      ('input', layers.InputLayer),      
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('output', layers.DenseLayer),
	],
    # layer parameters:
    input_shape=(None,93),  # 96x96 input pixels per batch

    hidden1_num_units=800,  # number of units in hidden layer
    hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    dropout1_p=0.5,

    hidden2_num_units=800,
    hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    dropout2_p=0.5,

    hidden3_num_units=800,

    #hidden5_nonlinearity=nonlinearities.rectify,
    #dropout5_p=0.5,

    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    objective=L2Regularization,
    objective_alpha=1E-9,

    eval_size=0.0,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
	AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
	#AdjustVariable('update_momentum', start=0.9, stop=0.999),
	#EarlyStopping(patience=200),
	],


    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=50,  # we want to train this many epochs
    verbose=1,
    )

nnet1 = NeuralNet(
    layers=[ 
	('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('maxout1', Maxout),
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('maxout2', Maxout),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    # layer parameters:
    input_shape=(None,93),

    hidden1_num_units=600,  # number of units in hidden layer 
    hidden1_nonlinearity=None,
    dropout1_p=0.5,
    maxout1_ds=2,
    
    hidden2_num_units=600, #300
    hidden2_nonlinearity=None,
    dropout2_p=0.0,
    maxout2_ds=2,
    
    hidden3_num_units=600,
    
    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    eval_size=0.2,

    #objective=L2Regularization,
    #objective_alpha=0.0001,
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),

    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=50,  # we want to train this many epochs
    verbose=1,
    
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        #EarlyStopping(patience=20),
        ],  
    )