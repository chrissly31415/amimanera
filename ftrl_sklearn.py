#!/usr/bin/python
'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''

import sys
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt





##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D=2**10, interaction=False, maxiter=10, holdout=5, roundp=5):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # iterations
        self.maxiter = maxiter
        self.holdout = holdout
        self.roundp = roundp

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y
        #print("g:%5.2f p:%5.2f y:%5.2f"%(g,p,y))

        #print "x:",x
        # update z and n for each feature i
        for i in self._indices(x):
            #print("i: %r:"%i)
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]   # weights for each feature i
            n[i] += g * g   #  sum of gradient squared, save for each feature i
            #print "n:",n


    def fit(self,train):
        """
        Fit data via sklearn train function

        """
        # start training
        for e in xrange(self.maxiter):
            loss = 0.
            count = 1

            for t, ID, x, y in data(train, self.D, self.roundp):  # data is a generator
                #    t: just a instance counter
                # date: you know what this is
                #   ID: id provided in original data
                #    x: features
                #    y: label (click)

                # step 1, get prediction from learner
                p = learner.predict(x)
                #print p
                #raw_input()

                if self.holdout and t % self.holdout == 0:
                    # step 2-1, calculate validation loss
                    #           we do not train with the validation data so that our
                    #           validation loss is an accurate estimation
                    #
                    # holdout: validate with every N instance, train with others
                    loss += logloss(p, y)
                    count += 1

                else:
                    # step 2-2, update learner with label (click) information
                    learner.update(x, p, y)

            print('Iteration %d finished, validation logloss: %f, elapsed time: %s' % (
                e, loss/count, str(datetime.now() - start)))
            sys.stdout.flush()



def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D, roundp):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    # iterate over file
    for t, row in enumerate(DictReader(open(path))):
        # process id

        if 't_id' in row.keys():
            ID = row['t_id']
        else:
            ID = t

        # process clicks
        y = 0.
        target = 'target'
        if target in row:
            if row[target] == '1':
                y = 1.
            del row[target]

        # extract date
        # turn hour really into hour, it was originally YYMMDDHH
        #row['hour'] = row['hour'][6:]

        # build x
        x = []
        for key in row:
            value = row[key]
            value = str(round(float(value),roundp))
            # round here ??
            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)
        yield t, ID, x, y


if __name__ == "__main__":
    """
    MAIN PART
    """
    ##############################################################################
    # parameters #################################################################
    ##############################################################################

    # plot with gnuplot -noraise ftrl.plt

    # A, paths
    train = '/home/loschen/Desktop/datamining-kaggle/numerai/data/numerai_training_data.csv'               # path to training file
    test = '/home/loschen/Desktop/datamining-kaggle/numerai/data/numerai_tournament_data.csv'                 # path to testing file
    submission = 'submissions/numerai_ftrl_29072016.csv'  # path of to be outputted submission file

    # B, model
    alpha = .0001  # learning rate
    beta = alpha   # smoothing parameter for adaptive learning rate
    L1 = 100.     # L1 regularization, larger value means more regularized
    L2 = 10.     # iL2 regularization, larger value means more regularized

    # C, feature/hash trick
    D = 4000     # number of hashed features to use # 20000
    interaction = True     # whether to enable poly2 feature interactions
    roundp = 1

    # D, training/validation
    maxiter = 60       # learn training data for N passes
    holdout = 5  # use every N training instance for holdout validation

    ##############################################################################
    # start training #############################################################
    ##############################################################################

    start = datetime.now()

    # initialize ourselves a learner
    learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction, maxiter, holdout, roundp)
    learner.fit(train)

    ##############################################################################
    # start testing, and build Kaggle's submission file ##########################
    ##############################################################################

    with open(submission, 'w') as outfile:
        outfile.write('t_id,probability\n')
        for t, ID, x, y in data(test, D, roundp):
            p = learner.predict(x)
            outfile.write('%s,%s\n' % (ID, str(p)))
