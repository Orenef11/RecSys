"""
Bayesian Personalized Ranking
Matrix Factorization model and a variety of classes
implementing different sampling strategies.
"""

import numpy as np
from math import exp, sin, cos
import random
import datetime

class BPRArgs(object):

    def __init__(self,learning_rate=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors

class BPR(object):

    def __init__(self,D,args, sins):
        """initialise BPR matrix factorization model
        D: number of factors
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors
        self.sins = sins

    def train(self,data,sampler,num_iters):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        """
        self.init(data)

        print ('initial loss = {0}'.format(self.loss()))
        for it in range(num_iters):
            print( 'starting iteration {0}'.format(it))
            for u,i,j,ui_time in sampler.generate_samples(self.data):
                self.update_factors(u,i,j,ui_time)
            print ('iteration {0}: loss = {1}'.format(it,self.loss()))

    def init(self,data):
        self.data = data
        self.num_users,self.num_items = self.data.shape

        self.item_bias = np.zeros(self.num_items)
        self.user_factors = np.random.random_sample((self.num_users,self.D))
        self.item_factors = np.random.random_sample((self.num_items,self.D))
        self.start_time = self.get_start_time()
        self.end_time = self.get_end_time()
        self.item_time_influence_elevation = np.random.random_sample((self.num_items,1))
        self.item_time_influence_sin_amplitude = []
        self.item_time_influence_sin_wave_length = []
        self.item_time_influence_sin_phase = []
        for i in range(self.sins):
            self.item_time_influence_sin_amplitude.append(np.random.random_sample((self.num_items,1)))
            self.item_time_influence_sin_wave_length.append(np.random.random_sample((self.num_items,1)))
            self.item_time_influence_sin_phase.append(np.random.random_sample((self.num_items,1)))

        self.create_loss_samples()

    def get_start_time(self):
        return int(self.data.data.min())

    def get_end_time(self):
        return int(self.data.data.max())

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_users**0.5)

        print( 'sampling {0} <user,item i,item j> triples...'.format(num_loss_samples))
        sampler = UniformUserUniformItem(True)
        self.loss_samples = [t for t in sampler.generate_samples(data,num_loss_samples)]

    @staticmethod
    def sig(x):
        if x < 709:
            return 0
        return 1.0 / (1.0 + exp(-x))

    def get_time_diff(self, ui_time):
        # cast to [-1, 1]
        return int(2 * (ui_time - self.start_time) / (self.end_time - self.start_time) - 1)

    def update_factors(self,u,i,j,ui_time,update_u=True,update_i=True):
        """apply SGD update"""
        update_j = self.update_negative_item_factors
        time_diff = self.get_time_diff(ui_time)

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:]) \
            + (self.item_time_influence_elevation[i] - self.item_time_influence_elevation[j]) * time_diff
	    
        for k in range(self.sins):
            x += self.item_time_influence_sin_amplitude[k][i] * sin(self.item_time_influence_sin_wave_length[k][i] * time_diff + self.item_time_influence_sin_phase[k][i])
        
        z = self.sig(x)

        # update bias terms
        if update_i:
            d = z * (1 - z) - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z * (1 - z) - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:]-self.item_factors[j,:]) * z * (1 - z) - self.user_regularization*self.user_factors[u,:]
            self.user_factors[u,:] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u,:] * z * (1 - z) - self.positive_item_regularization*self.item_factors[i,:]
            self.item_factors[i,:] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u,:] * z * (1 - z) - self.negative_item_regularization*self.item_factors[j,:]
            self.item_factors[j,:] += self.learning_rate*d
 
        # update item_time_influence_elevation:
        d = z * (1 - z) * time_diff - self.bias_regularization * self.item_time_influence_elevation[i]
        self.item_time_influence_elevation[i] += self.learning_rate * d
        d = -z * (1 - z) * time_diff - self.bias_regularization * self.item_time_influence_elevation[j]
        self.item_time_influence_elevation[j] += self.learning_rate * d

        for k in range(self.sins):
            # update item_time_influence_sin_amplitude:
            d = z * (1 - z) * sin(self.item_time_influence_sin_wave_length[k][i] * time_diff + self.item_time_influence_sin_phase[k][i]) - self.bias_regularization * self.item_time_influence_sin_amplitude[k][i]
            self.item_time_influence_sin_amplitude[k][i] += self.learning_rate * d
            d = -z * (1 - z) * sin(self.item_time_influence_sin_wave_length[k][j] * time_diff + self.item_time_influence_sin_phase[k][j]) - self.bias_regularization * self.item_time_influence_sin_amplitude[k][j]
            self.item_time_influence_sin_amplitude[k][j] += self.learning_rate * d
            
            # update item_time_influence_sin_wave_length:
            d = z * (1 - z) * self.item_time_influence_sin_amplitude[k][i] * time_diff * cos(self.item_time_influence_sin_wave_length[k][i] * time_diff + self.item_time_influence_sin_phase[k][i]) - self.bias_regularization * self.item_time_influence_sin_wave_length[k][i]
            self.item_time_influence_sin_wave_length[k][i] += self.learning_rate * d
            d = -z * (1 - z) * self.item_time_influence_sin_amplitude[k][j] * time_diff * cos(self.item_time_influence_sin_wave_length[k][j] * time_diff + self.item_time_influence_sin_phase[k][j]) - self.bias_regularization * self.item_time_influence_sin_wave_length[k][j]
            self.item_time_influence_sin_wave_length[k][j] += self.learning_rate * d
            
            # update item_time_influence_sin_phase:
            d = z * (1 - z) * self.item_time_influence_sin_amplitude[k][i] * cos(self.item_time_influence_sin_wave_length[k][i] * time_diff + self.item_time_influence_sin_phase[k][i]) - self.bias_regularization * self.item_time_influence_sin_phase[k][i]
            self.item_time_influence_sin_phase[k][i] += self.learning_rate * d
            d = -z * (1 - z) * self.item_time_influence_sin_amplitude[k][j] * cos(self.item_time_influence_sin_wave_length[k][j] * time_diff + self.item_time_influence_sin_phase[k][j]) - self.bias_regularization * self.item_time_influence_sin_phase[k][j]
            self.item_time_influence_sin_phase[k][j] += self.learning_rate * d

    def loss(self):
        ranking_loss = 0;
        for u,i,j,ui_time in self.loss_samples:
            x = self.predict(u,i, ui_time) - self.predict(u,j, ui_time)
            ranking_loss += self.sig(x)

        complexity = 0;
        for u,i,j,ui_time in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2

        return ranking_loss + 0.5*complexity

    def predict(self,u,i, ui_time):
        time_diff = self.get_time_diff(ui_time)
        x = self.item_bias[i] + np.dot(self.user_factors[u],self.item_factors[i]) \
            + int(self.item_time_influence_elevation[i]) * time_diff
        for k in range(self.sins):
            x += self.item_time_influence_sin_amplitude[k][i] * sin(self.item_time_influence_sin_wave_length[k][i] * time_diff + self.item_time_influence_sin_phase[k][i])
        return x


# sampling strategies

class Sampler(object):

    def __init__(self,sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self,data,max_samples=None):
        self.data = data
        self.num_users,self.num_items = data.shape
        self.max_samples = max_samples

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            while len(self.data[u].indices) == 0:
                u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0,self.num_items-1)
        return i

    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)

class UniformUserUniformItem(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            while len(self.data[u].indices) == 0:
                u = self.uniform_user()
            # sample positive item
            i = random.choice(self.data[u].indices)
            j = self.sample_negative_item(self.data[u].indices)
            yield u,i,j, data[u].getcol(i).data.item()

class UniformUserUniformItemWithoutReplacement(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(self,data,max_samples)
        # make a local copy of data as we're going to "forget" some entries
        self.local_data = self.data.copy()
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item without replacement if we can
            user_items = self.local_data[u].nonzero()[1]
            if len(user_items) == 0:
                # reset user data if it's all been sampled
                for ix in self.local_data[u].indices:
                    self.local_data[u,ix] = self.data[u,ix]
                user_items = self.local_data[u].nonzero()[1]
            i = random.choice(user_items)
            # forget this item so we don't sample it again for the same user
            self.local_data[u,i] = 0
            j = self.sample_negative_item(user_items)
            yield u,i,j, data[u].getcol(i).data.item()

class UniformPair(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        for _ in range(self.num_samples(self.data.nnz)):
            idx = random.randint(0,self.data.nnz-1)
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u].indices)
            yield u,i,j, data[u].getcol(i).data.item()

class UniformPairWithoutReplacement(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        idxs = [i for i in range(self.data.nnz)]
        random.shuffle(idxs)
        self.users,self.items = self.data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u].indices)
            self.idx += 1
            yield u,i,j, data[u].getcol(i).data.item()

class ExternalSchedule(Sampler):

    def __init__(self,filepath,index_offset=0):
        self.filepath = filepath
        self.index_offset = index_offset

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        f = open(self.filepath)
        samples = [map(int,line.strip().split()) for line in f]
        random.shuffle(samples)  # important!
        num_samples = self.num_samples(len(samples))
        for u,i,j in samples[:num_samples]:
            yield u-self.index_offset,i-self.index_offset,j-self.index_offset

if __name__ == '__main__':

    # learn a matrix factorization with BPR like this:

    import argparse
    import sys
    from scipy.io import mmread

    parser = argparse.ArgumentParser(description='BPR')
    parser.add_argument('-l', '--latent', dest='latent', default=10, type=int, help='Latent dimension')
    parser.add_argument('-r', '--learning-rate', dest='learning_rate', default=0.3, type=float, help='Learning rate')
    parser.add_argument('-s', '--sins', dest='sins', default=0, type=int, help='The amount of added sin functions')
    parser.add_argument('-e', '--epochs', dest='epochs', default=10, type=int, help='The amount of learning iterations')
    parser.add_argument('-pp', '--print-predictions', dest='print_predictions', action="store_true", help='Print the predictions table')
    parser.add_argument('-ps', '--print-sins', dest='print_sins', action="store_true", help='Print the sin functions cofficents')
    parser.add_argument('path')
    input_args = parser.parse_args()
    args = BPRArgs()
    
    data = mmread(input_args.path).tocsr()
    args.learning_rate = input_args.learning_rate

    model = BPR(input_args.latent,args, input_args.sins)

    sample_negative_items_empirically = True
    sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
    model.train(data,sampler,input_args.epochs)
    
    if input_args.print_predictions:
        predictions = []
        rows, cols = data.get_shape()
        for row in range(rows):
            prediction_row = []
            for col in range(cols):
                prediction_row.append(model.predict(row, col, datetime.datetime.utcnow().timestamp()))
            predictions.append(prediction_row)    
        print(predictions)
    
    if input_args.print_sins:
        rows, cols = data.get_shape()
        for i in range(cols):
            for k in range(model.sins):
                print('%s * Sin(%s * t + %s)' % (float(model.item_time_influence_sin_amplitude[k][i]), 
                      float(model.item_time_influence_sin_wave_length[k][i]), float(model.item_time_influence_sin_phase[k][i])))
            print()