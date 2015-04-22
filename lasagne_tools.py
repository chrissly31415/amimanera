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