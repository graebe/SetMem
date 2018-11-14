# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:00:29 2018

TODO
- prediction has to be filled in manually so far -> make model predictable

@author: P355139
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

class Dataset():
    
    def __init__(self,X,Y,t,dt=None,scale_X=None,scale_Y=None,name=None):
        
        self.name = name
        self.t = t
        self.dt = dt
        self.X = X
        self.Y = Y
        self.Y_pred = None
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.N = self.X.shape[0]
        self.n_u = self.X.shape[1]
        self.n_y = self.Y.shape[1]
    
    def plot_Y(self):
        plt.figure()
        plt.plot(self.Y)
        plt.grid()
        if self.Y_pred is not(None):
            plt.plot(self.t,self.Y_pred)
        if self.name is not(None):
            plt.title('Outputs Y of ' + self.name)
            
    def plot_X(self):
        plt.figure()
        plt.plot(self.t,self.X)
        plt.grid()
        if self.name is not(None):
            plt.title('Inputs X of ' + self.name)

class SMemPoints():
    
    def __init__(self,phi,v,t,name=None):
        
        self.name = name
        self.t = t
        self.phi = phi
        self.v = v

class Estimator_f_b():
    
    def __init__(self,f_b=None,pred_train=None,pred_val=None,pred_test=None):
        
        self.f_b=f_b
        self.pred_train=pred_train
        self.pred_val=pred_val
        self.pred_test=pred_test
        
#    def predict(self,X,N,m):
#        y_hat = []
#        for i in range(N-m):
#            k = i+m
#            y_hat.append(self.f_b.predict(self._phi_at_k(k,mode,residual)))
#        y_hat = np.stack(y_hat)
    
class SetMembershipEstimator():
    
    def __init__(self,data_train=None,data_test=None,data_val=None,SMemPoints=None,m=None,rho=None,gamma=None,epsilon=None,f_b=None):
        
        # Write Dataset
        self.v_train = data_train
        self.v_val = data_val
        self.v_test = data_test
        
        # Write Residual Data Set
        self.delta_v_train = None
        self.delta_v_val = None
        self.delta_v_test = None
        
        # Set Membership Data Subset
        self.SetMemPoints = SMemPoints
        
        # Predictor
        self.f_b = f_b
        
        # Parameters
        self.m = m
        self.rho = rho
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Predictions
        self.predictions = {'train':None,'val':None,'test':None}
    
#    def f_b_predict(self,mode=None,residual=False):
#        if residual:
#            if mode == 'train':
#                N = self.delta_v_train.N
#            elif mode == 'val':
#                N = self.delta_v_val.N
#            elif mode == 'test':
#                N = self.delta_v_test.N
#        else:
#            if mode == 'train':
#                N = self.v_train.N
#            elif mode == 'val':
#                N = self.v_val.N
#            elif mode == 'test':
#                N = self.v_test.N
    
    def get_regressor_dataset(self,mode=None,residual=False):
        if residual:
            print('not implemented yet')
        else:
            if residual:
                if mode == 'train':
                    N = self.delta_v_train.N
                    Y = self.delta_v_train.Y[self.m:]
                elif mode == 'val':
                    N = self.delta_v_val.N
                    Y = self.delta_v_val.Y[self.m:]
                elif mode == 'test':
                    N = self.delta_v_test.N
                    Y = self.delta_v_val.Y[self.m:]
            else:
                if mode == 'train':
                    N = self.v_train.N
                    Y = self.v_train.Y[self.m:]
                elif mode == 'val':
                    N = self.v_val.N
                    Y = self.v_val.Y[self.m:]
                elif mode == 'test':
                    N = self.v_test.N
                    Y = self.v_test.Y[self.m:]
        phi = []
        for i in range(N-self.m):
            k = i+self.m
            phi.append(self._phi_at_k(k,mode=mode))
        phi = np.stack(phi,axis=0)
        return phi, Y
    
    def _phi_at_k(self,k,mode=None,residual=False):
        if k<self.m:
            print('Warning: Cannot get phi at time step k=' +str(k)+ ', since window param m=' +str(self.m))
            regr_at_k = None
        else:
            if residual:
                if mode == 'train':
                    X = self.delta_v_train.X
                elif mode == 'val':
                    X = self.delta_v_val.X
                elif mode == 'test':
                    X = self.delta_v_test.X
            else:
                if mode == 'train':
                    X = self.v_train.X
                elif mode == 'val':
                    X = self.v_val.X
                elif mode == 'test':
                    X = self.v_test.X
            regr_at_k = X[k-self.m+1:k+1,:]
        return regr_at_k.flatten()
    
    def _v_at_k(self,k,mode=None,residual=False):
        if k<self.m:
            print('Warning: Cannot get phi at time step k=' +str(k)+ ', since window param m=' +str(self.m))
            v_k = None
        else:
            if residual:
                if mode == 'train':
                    Y = self.delta_v_train.Y
                elif mode == 'val':
                    Y = self.delta_v_val.Y
                elif mode == 'test':
                    Y = self.delta_v_test.Y
            else:
                if mode == 'train':
                    Y = self.v_train.Y
                elif mode == 'val':
                    Y = self.v_val.Y
                elif mode == 'test':
                    Y = self.v_test.Y
            v_k = Y[k,:]
        return v_k
    
    def _f_bounds(self,phi_tilde_t,mode):
        
        # Calculation Beta
        I = np.ones(self.delta_v_train.n_u)
        beta = np.diag(np.concatenate([I*self.rho**i for i in range(self.m)],axis=0))

        # Loop over all Set Membership Basis Points
        f_bounds_k_vect = []
        h_inf_vect = []
        for phi_tilde_k,v_tilde_k in zip(self.SetMemPoints.phi,self.SetMemPoints.v):
            
            # Difference between Set Membership Regressors and current Regressor
            phi_tilde_diff = np.expand_dims(phi_tilde_t-phi_tilde_k,axis=1)
            
            # H-infinity Norm of beta*regressor-difference
            beta_times_phi_tilde_diff = np.matmul(beta,phi_tilde_diff)
            h_inf = np.max(np.abs(beta_times_phi_tilde_diff))
            h_inf_vect.append(h_inf)
            
            # Calculate Upper Bound
            if mode == 'upper':
                f_bounds_k_curr = v_tilde_k + self.epsilon + self.gamma*h_inf
            if mode == 'lower':
                f_bounds_k_curr = v_tilde_k - self.epsilon - self.gamma*h_inf
            
            # f_upper
            f_bounds_k_vect.append(f_bounds_k_curr)
        
        # Take minimum
        if mode == 'upper':
            f_bounds_k = np.min(f_bounds_k_vect)
        if mode == 'lower':
            f_bounds_k = np.max(f_bounds_k_vect)
        
        # Return
        return f_bounds_k, np.squeeze(f_bounds_k_vect)
    
    def predict_data_set(self,mode=None):
        if mode=='train':
            X = self.delta_v_train.X
        elif mode=='val':
            X = self.delta_v_val.X
        elif mode=='test':
            X = self.delta_v_test.X
        N = X.shape[0]
        pred = []
        for i in tqdm(range(N-self.m)):
            k = i+self.m
            pred_curr = self.predict_SetMem(phi_tilde=np.expand_dims(self._phi_at_k(k,mode=mode,residual=True),axis=0))
            pred.append(pred_curr)
        pred = np.squeeze(np.stack(pred))
        self.predictions[mode] = pred
    
    def predict_data_set_parallel(self,mode=None):
        if mode=='train':
            X = self.delta_v_train.X
        elif mode=='val':
            X = self.delta_v_val.X
        elif mode=='test':
            X = self.delta_v_test.X
        N = X.shape[0]
        pool = multiprocessing.Pool()
        pred = pool.map_async(self._predict_par,range(N-self.m))
        pool.close()
        pool.join()
        pred = np.squeeze(np.stack(pred))
        self.predictions[mode] = pred
    
    def _predict_par(self,i,mode):
        k = i+self.m
        print(k)
        return self.predict_SetMem(phi_tilde=np.expand_dims(self._phi_at_k(k,mode=mode,residual=True),axis=0))
    
    def predict_SetMem(self,phi_tilde):
        # phi_tilde shape: (timesteps,regressors)
        f_lower = []
        f_upper = []
        f_pred  = []
        for phi_tilde_t in phi_tilde:
            f_lower.append(self._f_bounds(phi_tilde_t,mode='lower')[0])
            f_upper.append(self._f_bounds(phi_tilde_t,mode='upper')[0])
            f_pred.append(0.5*(f_lower[-1]+f_upper[-1]))
        f_lower = np.stack(f_lower,axis=0)
        f_upper = np.stack(f_upper,axis=0)
        f_pred  = np.stack(f_pred ,axis=0)
        return f_pred, f_lower, f_upper
    
    def predict(self,phi_tilde):
        print('Not implemented yet.')
        return
    
    def evaluate(self,phi_tilde,v_tilde):
        print('Not implemented yet.')
        return
    

    
    def generate_residual_dataset(self,perform_prediction=False):
        # Get Predictions of f_b
        if perform_prediction:
#            pred_train = self.f_b.predict(np.expand_dims(self.v_train.X))
#            pred_val = self.f_b.predict(self.v_val.X)
#            pred_test = self.f_b.predict(self.v_test.X)
            print('not implemented')
        # Write Resitual Data Sets
        self.delta_v_train = Dataset(X=self.v_train.X,Y=self.v_train.Y-self.f_b.pred_train)
        self.delta_v_val = Dataset(X=self.v_val.X,Y=self.v_val.Y-self.f_b.pred_val)
        self.delta_v_test = Dataset(X=self.v_test.X,Y=self.v_test.Y-self.f_b.pred_test)
        
    
    def plot_exponential_bound(self):
        m_ = np.arange(0,-self.m,-1)
        bound = np.array([self.rho**i for i in range(self.m)])
        plt.figure()
        plt.plot(m_,bound)
        plt.grid()
        plt.ylim([0,1])
        plt.ylabel('Exponential Bound')
        plt.xlabel('Past Timesteps [k]')
    
    def plot_fb_regression_plot(self):
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(self.v_train.Y[self.m+1:]*self.v_train.scale_Y)
        plt.plot(self.f_b.pred_train*self.v_train.scale_Y)
        plt.grid()
        plt.ylabel('Target')
        plt.title('Train')
        plt.subplot(3,1,2)
        plt.plot(self.v_val.Y[self.m+1:]*self.v_train.scale_Y)
        plt.plot(self.f_b.pred_val*self.v_train.scale_Y)
        plt.grid()
        plt.ylabel('Target')
        plt.title('Validation')
        plt.subplot(3,1,3)
        plt.plot(self.v_test.Y[self.m+1:]*self.v_train.scale_Y)
        plt.plot(self.f_b.pred_test*self.v_train.scale_Y)
        plt.grid()
        plt.ylabel('Target')
        plt.title('Test')
        plt.xlabel('Time t [s]')        

class SMemPoints_Picker():
    
    def __init__(self,Estimator=None):
        
        self.Estimator = Estimator
        
    def set_SMemPoints(self,ds=None,method='downsampling'):
        # Init with None
        phi_SMem, v_SMem, t_SMem = None, None, None
        
        if method == 'downsampling':
            phi_SMem = []
            v_SMem = []
            t_SMem = []
            for i in range(np.int((np.floor(self.Estimator.delta_v_train.N-self.Estimator.m)/ds))):
                k = i*ds+self.Estimator.m
                phi_SMem.append(self.Estimator._phi_at_k(k=k,mode='train',residual=True))
                v_SMem.append(self.Estimator._v_at_k(k=k,mode='train',residual=True))
                t_SMem.append(self.Estimator.delta_v_train.t[k])
            phi_SMem = np.stack(phi_SMem)
            v_SMem = np.stack(v_SMem)
            t_SMem = np.stack(t_SMem)

        # Write to Object
        self.Estimator.SetMemPoints = SMemPoints(phi=phi_SMem,v=v_SMem,t=t_SMem)
        
class SMemEst_FSS_Toolkit():
    
    def __init__(self,Estimator=None):
        
        self.Estimator = Estimator
        self.FSS_Analysis = None
        self.gamma_star_matr = None
        self.gamma_star = None
        
    def _check_if_f_upper_t_valid(self,t,indices=None,mode=None,residual=None):
        # Get Regressors and target
        v_tilde_t = self.Estimator._v_at_k(k=t,mode=mode,residual=residual)
        phi_tilde_t = self.Estimator._phi_at_k(k=t,mode=mode,residual=residual)
        # Get Upper Bound
        f_upper_t, f_upper_t_vect = self.Estimator._f_bounds(phi_tilde_t,mode='upper')
        # Check Condition
        valid = f_upper_t > v_tilde_t-self.Estimator.epsilon
        valid_vect = np.array(f_upper_t_vect) > v_tilde_t-self.Estimator.epsilon
        return valid, f_upper_t, valid_vect, f_upper_t_vect
    
    def check_if_FSS_is_not_empty(self,indices=None,verbose=False,mode=None,residual=False):
        # If no indices are picked, take all
        if residual:
            if mode=='train':
                N = self.Estimator.delta_v_train.N
            elif mode=='val':
                N = self.Estimator.delta_v_val.N
            elif mode=='test':
                N = self.Estimator.delta_v_test.N
        else:
            if mode=='train':
                N = self.Estimator.v_train.N
            elif mode=='val':
                N = self.Estimator.v_val.N
            elif mode=='test':
                N = self.Estimator.v_test.N
        if indices is None:
            indices = np.arange(0,N-self.Estimator.m)+self.Estimator.m
        # Iterate over indices
        valid_vect = []
        for t in indices:
            if verbose:
                print('Checking timestep ' + str(t))
            valid_list, _, _, _ = self._check_if_f_upper_t_valid(t=t,indices=indices,mode=mode,residual=residual)
            valid = valid_list[0]
            valid_vect.append([t,valid])
            if not(valid):
                if verbose:
                    print('   NOT valid!')
                return False
            else:
                if verbose:
                    print('   Valid.')
        return True
    
    def calculate_gamma_star(self,indices=None,mode=None,residual=None):
        # If no indices are picked, take all
        if residual:
            if mode=='train':
                N = self.Estimator.delta_v_train.N
                n_u = self.Estimator.delta_v_train.n_u
            elif mode=='val':
                N = self.Estimator.delta_v_val.N
                n_u = self.Estimator.delta_v_val.n_u
            elif mode=='test':
                N = self.Estimator.delta_v_test.N
                n_u = self.Estimator.delta_v_test.n_u
        else:
            if mode=='train':
                N = self.Estimator.v_train.N
                n_u = self.Estimator.v_train.n_u
            elif mode=='val':
                N = self.Estimator.v_val.N
                n_u = self.Estimator.v_val.n_u
            elif mode=='test':
                N = self.Estimator.v_test.N
                n_u = self.Estimator.v_test.n_u
        if indices is None:
            indices = np.arange(0,N-self.Estimator.m)+self.Estimator.m
        
        # Calculation Beta
        I = np.ones(n_u)
        beta = np.diag(np.concatenate([I*self.Estimator.rho**i for i in range(self.Estimator.m)],axis=0))
        
        # Initialize gamma star matrix
        gamma_star = np.zeros(shape=(indices.shape[0],indices.shape[0]),dtype=np.float)
        
        for i_t,t in tqdm(enumerate(indices)):
            
            for i_k,k in enumerate(indices):
                
                # Pass if comparing to self
                if i_t == i_k:
                    continue
                
                # Get Regressors and target
                v_tilde_t   = self.Estimator._v_at_k  (k=t,mode=mode,residual=residual)
                phi_tilde_t = self.Estimator._phi_at_k(k=t,mode=mode,residual=residual)
                v_tilde_k   = self.Estimator._v_at_k  (k=k,mode=mode,residual=residual)
                phi_tilde_k = self.Estimator._phi_at_k(k=k,mode=mode,residual=residual)
                
                # Calculate Denominator of gamma star
                phi_tilde_diff = np.expand_dims(phi_tilde_t-phi_tilde_k,axis=1)
                beta_times_phi_tilde_diff = np.matmul(beta,phi_tilde_diff)
                h_inf = np.max(np.abs(beta_times_phi_tilde_diff))
                
                # Calculate gamma star
                gamma_star_tk = (v_tilde_k-v_tilde_t+2*self.Estimator.epsilon) / h_inf
                
                # Write to Matrix
                gamma_star[i_t,i_k] = gamma_star_tk
        
        # Write to Object
        self.gamma_star_matr = gamma_star
        self.gamma_star = [np.max(gamma_star),np.unravel_index(gamma_star.argmax(), gamma_star.shape)]
        
    def plot_gamma_star(self):
        plt.figure()
        plt.imshow(self.gamma_star_matr, cmap='hot', interpolation='nearest')
        plt.show()
        plt.ylabel('Timestep t')
        plt.xlabel('Timestep k')
        plt.title('Maximum gamma*={:.3f} at t,k=[{:d},{:d}].'.format(self.gamma_star[0], self.gamma_star[1][0], self.gamma_star[1][1]))
    
    def generate_validity_vector_for_FSS(self,indices=None,verbose=False,mode=None,residual=False):
        # If no indices are picked, take all
        if residual:
            if mode=='train':
                N = self.Estimator.delta_v_train.N
            elif mode=='val':
                N = self.Estimator.delta_v_val.N
            elif mode=='test':
                N = self.Estimator.delta_v_test.N
        else:
            if mode=='train':
                N = self.Estimator.v_train.N
            elif mode=='val':
                N = self.Estimator.v_val.N
            elif mode=='test':
                N = self.Estimator.v_test.N
        if indices is None:
            indices = np.arange(0,N-self.Estimator.m)+self.Estimator.m
        # Iterate over indices
        valid_list = []
        valid_vect_list = []
        f_upper_list = []
        f_upper_vect_list = []
        for t in tqdm(indices):
            if verbose:
                print('Checking timestep ' + str(t))
            valid, f_upper_t, valid_vect, f_upper_t_vect = self._check_if_f_upper_t_valid(t=t,indices=indices,mode=mode,residual=residual)
            valid_list.append(valid)
            valid_vect_list.append(valid_vect)
            f_upper_list.append(f_upper_t)
            f_upper_vect_list.append(f_upper_t_vect)
            
        valid_list = np.stack(valid_list,axis=0)
        valid_vect_list = np.stack(valid_vect_list, axis=0)
        f_upper_list = np.stack(f_upper_list,axis=0)
        f_upper_vect_list = np.stack(f_upper_vect_list,axis=0)
        # Write to Object
        self.FSS_Analysis = {'mode':mode,'indices':indices,'valid_list':valid_list, 'valid_vect_list':valid_vect_list, 'f_upper_list':f_upper_list, 'f_upper_vect_list':f_upper_vect_list}
    
    def plot_FFS_validity(self):
    
#        idx_indices_valid = np.where(self.FSS_Analysis['valid_list'])[0]
#        idx_indices_not_valid = np.where(self.FSS_Analysis['valid_list']-1)[0]
    
        plt.figure()
        plt.subplot(1,1,1)
        plt.imshow(self.FSS_Analysis['valid_vect_list'], cmap='hot', interpolation='nearest')
        plt.show()
        plt.ylabel('Validity f_upper_tk')
        plt.xlabel('Timestep k')
#        plt.subplot(2,1,2)
#        plt.plot(targets,color='gray')
#        plt.plot(targets-epsilon)
#        plt.plot(indices[idx_indices_valid],f_upper_t[idx_indices_valid],'*',color='green')
#        plt.plot(indices[idx_indices_not_valid],f_upper_t[idx_indices_not_valid],'*',color='red')
#        plt.legend(['targets','FFS border','f_upper'])
#        plt.xlabel('Timestep k')
#        plt.grid()