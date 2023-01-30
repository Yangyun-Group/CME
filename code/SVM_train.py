# Copyright (c) YangYun Group Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Time    :   2023/01/30 19:28:20
Author  :   YangYun Group
Version :   1.0
Contact :   yangyun@nju.edu.cn
"""

import dataset
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pickle
import config
from sklearn.metrics import  r2_score, mean_squared_error
from predict import mean_absolute_error, POD
import json
import os
import argparse
import time
import sys
class SVM:
    def __init__(self,X_train,y_train,X_test,y_test, verbose=False):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.config = config.SVM_config
        self.verbose = verbose
        self.best_param = None
        self.min_loss = 100000


    def train(self):
        param_rbf = self.config['param_rbf']
        min_loss = self.min_loss
        for C_ in param_rbf['C']:
            for gamma_ in param_rbf['gamma']:
                svr_model = SVR(C=C_, gamma=gamma_,kernel='rbf')
                svr_model.fit(self.X_train, self.y_train)
                y_pre = svr_model.predict(self.X_test)
                loss = mean_absolute_error(self.y_test, y_pre)
                if loss < self.min_loss:
                    best_param = {'C':C_, 'gamma':gamma_, 'kernel':'rbf'}
                    self.min_loss = loss
                    if self.verbose:print(best_param,'min_loss:'+str(loss))
        param_poly = self.config['param_poly']
        for C_ in param_poly['C']:
            for gamma_ in param_poly['gamma']:
                for degree_ in param_poly['degree']:
                    for coef0_ in param_poly['coef0']:
                        svr_model = SVR(C=C_, gamma=gamma_,coef0=coef0_,degree=degree_, kernel='poly')
                        svr_model.fit(self.X_train, self.y_train)
                        y_pre = svr_model.predict(self.X_test)
                        loss = mean_absolute_error(self.y_test, y_pre)
                        if loss < self.min_loss:
                            best_param = {'C':C_, 'gamma':gamma_,'coef0':coef0_,'degree':degree_, 'kernel':'poly'}
                            self.min_loss = loss
                            if self.verbose:print(best_param,'min_loss:'+str(loss))
        param_sigmoid = self.config['param_sigmoid']
        for C_ in param_sigmoid['C']:
            for gamma_ in param_sigmoid['gamma']:
                for degree in param_sigmoid['degree']:
                    svr_model = SVR(C=C_, gamma=gamma_,degree=degree_,kernel='sigmoid')
                    svr_model.fit(self.X_train, self.y_train)
                    y_pre = svr_model.predict(self.X_test)
                    loss = mean_absolute_error(self.y_test, y_pre)
                    if loss < self.min_loss:
                        best_param = {'C':C_, 'gamma':gamma_,'degree':degree_, 'kernel':'sigmoid'}
                        self.min_loss = loss
                        if self.verbose:print(best_param,'min_loss:'+str(loss))
        self.best_param = best_param
        self.__load_model()
    
    def __load_model(self, best_param=None):
        if best_param == None:
            self.model = SVR(**self.best_param)
        else:
            self.model = SVR(**best_param)
        self.model.fit(self.X_train, self.y_train)
    
    def bulid_model(self,best_param=None):
        self.__load_model(best_param)
        
    def predict(self, X):
        return self.model.predict(np.array(X))

def train(args):
    my_data = dataset.DataSet(args.input_file)
    X, y = my_data.inputdata, my_data.y_data
    min_loss = 1000000
    best_param, best_random_seed = None, None
    save_epoch = args.train_epochs // 10
    for random_seed in range(args.train_epochs):
        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.split_dataset_percentage, random_state=random_seed)
        svm_model = SVM(X_train,y_train,X_test,y_test) 
        svm_model.train()
        y_pre = svm_model.model.predict(X_test)
        mae, r2, mse, pod_6 = mean_absolute_error(y_test, y_pre), r2_score(y_test, y_pre), mean_squared_error(y_test, y_pre), POD(y_test, y_pre, 5.9)
        pod = POD(y_test, y_pre, mae)
        print("epochs [{}]:time_cost: {}s, best_param:{}, mean_absolute_error: {}, r2_score: {}, mean_squared_error: {}, POD_5.9: {}, POD: {} ".format(random_seed, 
                int(time.time()-start),  svm_model.best_param, mae, r2, mse, pod_6, pod, file=sys.stderr))
        if svm_model.min_loss < min_loss:
           best_param, best_random_seed = svm_model.best_param, random_seed
        svm_model.bulid_model()
        if (random_seed+1) % save_epoch == 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.split_dataset_percentage, random_state=best_random_seed)
            svm_model = SVM(X_train, y_train, X_test, y_test)
            svm_model.bulid_model(best_param)
            with open(args.model_save_path+'_'+str(random_seed+1),'wb') as f:
                pickle.dump(svm_model.model, f)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./data/cme_final_soho.json')
    parser.add_argument('--mode', type=str, default='Wind After')
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--split_dataset_percentage', type=float, default=0.2, help='The value range of split_dataset_percentage: [0,1]')
    parser.add_argument('--model_save_path', type=str, default='./model/SVM_model')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
