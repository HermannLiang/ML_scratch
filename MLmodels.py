# implementation
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from numpy.linalg import det, lstsq
from scipy.linalg import cholesky, cho_solve
import seaborn as sns


def score(y_pred,y_true):
    ## implementation of R2 score
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    u = ((y_pred - y_true)**2).sum()
    v = ((y_true - y_true.mean())**2).sum()
    
    return 1-u/v

def rmsloss(y_pred,y_true):
    return np.sqrt(((y_pred - y_true)**2).mean())
def mseloss(y_pred,y_true):
    return ((y_pred - y_true)**2).mean()
def smseloss(y_pred,y_true):
    se = (y_pred - y_true)**2
    return (se/np.linalg.norm(se)).mean()

def toy_problem(variable_length, sample_length):
    
    X = np.empty(shape = (sample_length,1))
    full_X = 2*2*np.pi*np.arange(-variable_length/2,variable_length/2)/variable_length
    full_y = np.sin(full_X) + full_X/2
    if variable_length != sample_length:
        sample_idx = np.random.choice(variable_length,sample_length,replace = False)
        sample_idx = np.sort(sample_idx)
        X[:,0] = full_X[sample_idx]
        y = full_y[sample_idx]
        y-=y.mean()
        return X,y
    else:
        return full_X[:,np.newaxis],full_y-full_y.mean()

class NNRegressor():
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X_test,distance='eucli',k = 2):
        y_pred = np.zeros(shape=(X_test.shape[0],1))
        
        for i,sample in enumerate(X_test):
            if distance == 'eucli':
                dist = ((self.X - sample)**2).sum(axis = 1)
            elif distance == 'manh':
                dist = (self.X - sample).sum(axis = 1)
            if k==1:
                idx = np.argmin(dist)
                y_pred[i] = self.y[idx]
            else:
                idx = []
                for kk in range(k):
                    idx_ = np.argmin(dist)
                    dist[idx_] = np.inf
                    idx.append(idx_)
                y_pred[i] = self.y[idx].mean()

            # return indices of k mininal distances
            # set prediction to the average of y[idx]
        return y_pred.squeeze()   
    
class _LinearRegressor():

    @staticmethod
    def least_squares_error(x, y, w, c):
        squared_error = (np.square(y-w*x.squeeze()-c)).sum()
        return squared_error
    @staticmethod
    def calc_gradients(x, y, w, c):
        grad_w = np.sum(2*x*(w*x.squeeze()+c-y))
        grad_c = np.sum(2*(w*x.squeeze()+c-y))
        return grad_w, grad_c
    def run_iteration(self,x_train, y_train, w_current, c_current, E_current, 
                                      current_step_size, converge_threshold):

        # Set to True when converged..
        converged = False

        # evaluate the gradients..
        w_grad, c_grad = self.calc_gradients(x_train, 
                                             y_train,
                                             w_current,
                                             c_current)
        # gradient proportional to the step size..
        w_new = w_current-current_step_size*w_grad
        c_new = c_current-current_step_size*c_grad

        # evaluate and remember the squared error..
        E_new = self.least_squares_error(x_train, y_train, 
                                    w_new, c_new)

        if E_new>E_current:
            # reduce half the stepsize
            current_step_size = current_step_size*0.5


        # terminate the loop if converged..
        if E_new<=converge_threshold:
            converged = True

        # Take the step
        w_current = w_new
        c_current = c_new
        E_current = E_new

        return w_current, c_current, E_current, current_step_size, converged
                    
    def fit(self,X_train,y_train,num_iterations = 500,verbose = True,
            current_step_size = 0.001,
            converge_threshold = 1e-8,
            w_current = 1.5,
            c_current = 1):
        try:
            E_current = self.least_squares_error(X_train, y_train, w_current, c_current)
            for iteration in range(1,num_iterations+1):
                
                w_current, c_current, E_current, current_step_size, converged = \
                    self.run_iteration(X_train, y_train, w_current, c_current, E_current, 
                                  current_step_size, converge_threshold)
                if ((iteration)%int(num_iterations/10)==0) and verbose == True:
                    print('iteration %4d, E = %f, w = %f, c = %f' % 
                          (iteration, E_current, w_current, c_current))

                if converged:
                    # Break out of iteration loop..
                    print('Converged!')
                    break

            print('\nAfter gradient descent optimisation:')
            print('Optimised w = ', w_current)
            print('Optimised c = ', c_current)
            
            self.params = {'weights':w_current, 'constant':c_current}

        except Exception as err:
                print('Error during fit():', err)
        
        return self
    
    def predict(self,X_test):
        self.y_pred = (np.dot(X_test, np.array(self.params['weights'])) \
                      + self.params['constant']).squeeze()
        return self.y_pred
    
    
class LinearRegressor():

    @staticmethod
    def least_squares_error(x, y, w, c):
        squared_error = (np.square(y-np.einsum('nm,m->n', x, w)-c)).sum()
        return squared_error
    @staticmethod
    def calc_gradients(x, y, w, c):
        grad_w = 2*x.T.dot(np.einsum('nm,m->n', x, w)+c-y)//len(y)
        grad_c = 2*(np.einsum('nm,m->n', x, w)+c-y).sum()/len(y)
        return grad_w, grad_c
    def run_iteration(self,x_train, y_train, w_current, c_current, E_current, 
                                      current_step_size, converge_threshold):

        # Set to True when converged..
        converged = False

        # evaluate the gradients..
        w_grad, c_grad = self.calc_gradients(x_train, 
                                             y_train,
                                             w_current,
                                             c_current)
        # gradient proportional to the step size..
        w_new = w_current-current_step_size*w_grad
        c_new = c_current-current_step_size*c_grad

        # evaluate and remember the squared error..
        E_new = self.least_squares_error(x_train, y_train, 
                                    w_new, c_new)

        if E_new>E_current:
            # reduce half the stepsize
            current_step_size = current_step_size*0.9


        # terminate the loop if converged..
        if E_new<=converge_threshold:
            converged = True

        # Take the step
        w_current = w_new
        c_current = c_new
        E_current = E_new

        return w_current, c_current, E_current, current_step_size, converged
                    
    def fit(self,X_train,y_train,num_iterations = 500,verbose = True,
            current_step_size = 0.001,
            converge_threshold = 1e-8,
            w_current = None,
            c_current = None):
        
        if w_current is None:
            w_current = np.random.randn(X_train.shape[1])
        if c_current is None:
            c_current = np.random.randn(1)
            
            
        E_current = self.least_squares_error(X_train, y_train, w_current, c_current)
        for iteration in range(1,num_iterations+1):

            w_current, c_current, E_current, current_step_size, converged = \
                self.run_iteration(X_train, y_train, w_current, c_current, E_current, 
                              current_step_size, converge_threshold)
            self.current_step_size = current_step_size
            if ((iteration)%int(num_iterations/10)==0) and verbose == True:
                print('iteration %4d, E = %f' % 
                      (iteration, E_current))

            if converged:
                # Break out of iteration loop..
                print('Converged!')
                break

#         print('\nAfter gradient descent optimisation:')
#         print('Optimised w = ', w_current)
#         print('Optimised c = ', c_current)

        self.params = {'weights':w_current, 'constant':c_current}
        return self
    
    def fit_predict(self,X_train,y_train,X_test,y_test,
                    num_iterations = 500,verbose = False,
            current_step_size = 0.001,
            converge_threshold = 1e-8,
            w_current = None,
            c_current = None):
        """compute training and testing r2 scores while fitting,
           helpful for tuning parameters
        """
        self.train_mse = []
        self.test_mse = []
        if w_current is None:
            w_current = np.random.randn(X_train.shape[1])
        if c_current is None:
            c_current = np.random.randn(1)
        E_current = self.least_squares_error(X_train, y_train, w_current, c_current)
        num_iterations+=1
        pitstops = np.arange(0,num_iterations,100)
        for iteration in range(num_iterations):
            w_current, c_current, E_current, current_step_size, converged = \
                self.run_iteration(X_train, y_train, w_current, c_current, E_current, 
                              current_step_size, converge_threshold)
            self.current_step_size = current_step_size
            self.params = {'weights':w_current, 'constant':c_current}
            if iteration in pitstops:
                self.train_mse.append(mseloss(self.predict(X_train),y_train))
                self.test_mse.append(mseloss(self.predict(X_test),y_test))
                
            if converged:
                # Break out of iteration loop..
                print('Converged!')
                break

        
        return self
    
    def predict(self,X_test):
        self.y_pred = (np.dot(X_test, self.params['weights']) \
                      + self.params['constant']).squeeze()
        return self.y_pred
    
class Node:
    def __init__(self,x,y,idxs,
                 min_samples_split = 5,
                 max_depth = 5,
                 current_depth = 1):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = np.mean(y[idxs]) # this is the prediction made by this node
        self.score = float('inf') # this will init the score, score only will be overide if it splits
#         try:
        self.grow_tree()
#         except Exception as e:
#             print('current depth: ', current_depth)
#             print(e)
        
    def grow_tree(self):
        
            
        if (self.current_depth >= self.max_depth) or (len(self.idxs) < self.min_samples_split):
            return
        for col in range(self.col_count):
            self.find_split(col)
        if self.is_leaf:
            return
        x = self.split_col
        i_lhs = np.nonzero(x <= self.split)[0]
        i_rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(self.x, self.y,self.idxs[i_lhs],
                       min_samples_split = self.min_samples_split,
                       max_depth = self.max_depth,
                       current_depth = self.current_depth + 1)
        self.rhs = Node(self.x, self.y,self.idxs[i_rhs],
                       min_samples_split = self.min_samples_split,
                       max_depth = self.max_depth,
                       current_depth = self.current_depth + 1)
        
    def find_split(self,col_idx):
        x = self.x[self.idxs,col_idx]
        for r in range(self.row_count):
            i_lhs = x<= x[r]
            i_rhs = x>x[r]
            if i_lhs.sum()>0 and i_rhs.sum()>0: # to ensure no (0/all) split
#             if i_lhs.sum() > self.min_samples_split or i_rhs.sum() > self.min_samples_split:
                curr_score = self.find_score(i_lhs,i_rhs)
                if curr_score < self.score:
                    self.col_idx = col_idx
                    self.score = curr_score
                    self.split = x[r]
                    
#     def find_score(self,i_lhs,i_rhs):
# #         print(self.idxs.dtype)
#         i_lhs = np.nonzero(i_lhs)[0]
#         i_rhs = np.nonzero(i_rhs)[0]
#         y = self.y[self.idxs]
#         lhs_var = y[i_lhs].var()
#         rhs_var = y[i_rhs].var()
#         return lhs_var + rhs_var
    
    def find_score(self, lhs, rhs):
        y = self.y[self.idxs]
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()
    
    @property
    def split_col(self):
        """
        return a 1-d array with the column to be split
        """
        return self.x[self.idxs,self.col_idx]
    @property
    def is_leaf(self):
        return self.score == float('inf')
    def predict(self,x):
        return np.array([self.predict_row(x_row) for x_row in x])
    def predict_row(self,x_row):
        if self.is_leaf:
            return self.val
        node = self.lhs if x_row[self.col_idx]<= self.split else self.rhs
        return node.predict_row(x_row)
        
        
class DecisionTreeRegressor():
    
    def fit(self,X,y,
                 max_depth = 7,
                 min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = Node(X,y,
                         np.array(np.arange(len(y))),
                         min_samples_split = self.min_samples_split,
                         max_depth = self.max_depth)
#         self.tree = Node(X, y, np.array(np.arange(len(y))), 
#                          min_samples_split = min_samples_split)
        return self
    @staticmethod
    def mse(y):
        """
        compute the mean squared error about the mean
        """
        return (np.square(y-y.mean())).mean()
    
    def predict(self,X):
        return self.tree.predict(X).squeeze()

class RandomForestRegressor():
    
    def __init__(self,n_estimators,max_depth=5,min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.base_estimators = []
        self.training_preds = np.zeros(shape = (len(y),self.n_estimators))
        for i in range(self.n_estimators):
            bag_idx = np.random.choice(np.array(range(len(X))),size = len(X), replace = True)
            reg = DecisionTreeRegressor().fit(X[bag_idx,:],y[bag_idx],
                                            max_depth = self.max_depth,
                                            min_samples_split = self.min_samples_split)
            self.training_preds[:,i] = reg.predict(X[bag_idx,:])
            self.base_estimators.append(reg)
    def predict(self,X_test):
        self.testing_preds = np.zeros(shape = (len(X_test),self.n_estimators))
        for i,reg in enumerate(self.base_estimators):
            self.testing_preds[:,i] = reg.predict(X_test)
            
        return self.testing_preds.mean(axis = 1).squeeze()
    

def RBFkernel(X,Y,length_scale = 1.0,sigma_f = 1.0):
#     if type(X) != np.ndarray:
#         X = np.array(X)
#         Y = np.array(Y)
#     return sigma_f**2*np.exp(-0.5*np.subtract.outer(X/length_scale,Y/length_scale)**2).squeeze()

    sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(Y**2, 1) - 2 * np.dot(X, Y.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

def min_func(X_train,y_train,noise = 1e-8):
    # return the minimiztion objective function

    def compute_lml(params):
    # compute the log marginal likelihood
    # described in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
    # 2.2, Algorithm 2.1.
        K = RBFkernel(X_train, X_train, params[0], params[1]) + \
            noise**2 * np.eye(len(X_train))
        if not np.all(np.linalg.eigvals(K)>0):
            # add a tiny identity matrix for safety
            K += 1e-8*np.eye(len(X_train))

#         L = np.linalg.cholesky(K)
#         alpha = lstsq(L.T, lstsq(L, y_train)[0])[0]
        L = cholesky(K,lower = True)
        alpha = cho_solve((L,True),y_train)
        
        
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * y_train.T.dot(alpha) + \
               0.5 * len(X_train) * np.log(2*np.pi)

    return compute_lml


class GaussianProcessRegressor():
    
    def __init__(self,kernel='rbf'):
        
        if kernel == 'rbf':
            self.kernel = RBFkernel
            self.l = 1.0
            self.sigma_f = 1.0
            self.sigma_n = 1e-8

    def predict_prior(self,X):
        
        # Mean and covariance of the prior
        self.mean = np.zeros(X.shape)
        self.cov = self.kernel(X, X)
        
        return self.mean,self.cov
    def sample_y(self,X,n_samples = 4):
        
        # Draw samples from the prior
        samples = np.random.multivariate_normal(self.mean.ravel(), self.cov, n_samples)
        return samples
    def predict(self,X_train,y_train,X_test,sigma_n = 1e-8,
                             inv_method = 'cholesky'):
        self.sigma_n = sigma_n
        K_trtr = self.kernel(X_train,X_train,self.l,self.sigma_f) \
                 + self.sigma_n**2 * np.eye(len(X_train))
        K_trte = self.kernel(X_train,X_test,self.l,self.sigma_f)
        K_tete = self.kernel(X_test,X_test,self.l,self.sigma_f) \
                 + 1e-8 * np.eye(len(X_test))
        
        if inv_method == 'cholesky':
            # cholesky approach
            if not np.all(np.linalg.eigvals(K_trtr)>0):
                # add a tiny identity matrix for safety
                print('K_trtr is not symmetric positive-semidefinite.')
                K_trtr += 1e-8*np.eye(len(X_train))

            L = cholesky(K_trtr,lower = True)
            alpha = cho_solve((L,True),y_train)
            mean = K_trte.T.dot(alpha)
            v = cho_solve((L,True),K_trte)
            cov = K_tete - K_trte.T.dot(v)
            if not np.all(np.linalg.eigvals(cov)>0):
                print('predicted covariance is not symmetric positive-semidefinite.')
                cov+=1e-8*np.eye(len(cov))
                
        elif inv_method == 'normal':
    #     # normal

            K_inv = np.linalg.inv(K_trtr)
            mean = K_trte.T.dot(K_inv).dot(y_train)
            cov = K_tete - K_trte.T.dot(K_inv).dot(K_trte)

        self.mean = mean
        self.cov = cov
        
        return mean,cov
    
    def fit(self,X_train,y_train,sigma_n = 1e-8):
        
        self.X_train = X_train
        self.y_train = y_train
        self.sigma_n = sigma_n
        res = minimize(min_func(X_train, y_train, self.sigma_n), [self.l,self.sigma_f], 
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B')
        
        self.l, self.sigma_f = res.x 
        return self.l, self.sigma_f
    def log_marginal_likelihood(self):
        f = min_func(self.X_train,self.y_train,self.sigma_n)
        
        return -f([self.l,self.sigma_f]).ravel()[0]

    
        
                                    
