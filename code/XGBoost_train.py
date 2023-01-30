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
Time    :   2023/01/30 19:28:38
Author  :   YangYun Group
Version :   1.0
Contact :   yangyun@nju.edu.cn
"""

import dataset
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import config_shuffle as config
from sklearn.metrics import r2_score, mean_squared_error
from predict import POD, mean_absolute_error
import json
from xgboost.sklearn import XGBRegressor
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import warnings

class XGB_Tree:
    def __init__(self, X_train, y_train, X_test, y_test, verbose=False):
        """
        X_train, y_train, X_test, y_test : np.array, float64
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.config = config.XGB_Tree_config
        self.verbose = verbose
        self.eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        self.best_param = self.config['default_param'].copy()
        self.min_loss = 100000
    

    def train(self):
        # search best num of estimator rough and precise

        # print('search best num of estimator rough and precise')
        n_estimators_search = self.config['n_estimators']
        self.__search_one_param('n_estimators', n_estimators_search)

        # search best max_depth and min_child_weight
        # print('search best max_depth and min_child_weight')
        max_depth_search, min_child_weight_search = self.config['max_depth'], self.config['min_child_weight']
        self.__search_pair_param('max_depth', max_depth_search, 'min_child_weight', min_child_weight_search)

        # search best value of gamma 
        # print('search best value of gamma ')
        gamma_search = self.config['gamma']
        self.__search_one_param('gamma', gamma_search)

        # search best subsample and colsample_bytree
        # print('search best subsample and colsample_bytree ')
        subsample_search, colsample_search = self.config['subsample'],self.config['colsample_bytree']
        self.__search_pair_param('subsample', subsample_search, 'colsample_bytree', colsample_search)
        # self.__search_one_param('subsample', subsample_search)
        # self.__search_one_param('colsample_bytree', colsample_search)

        
        # search best reg_alpha and reg_lambda
        # print('search best reg_alpha and reg_lambda')
        alpha_search, lambda_search = self.config['reg_alpha'],self.config['reg_lambda']
        self.__search_pair_param('reg_alpha', alpha_search, 'reg_lambda', lambda_search)
        # self.__search_one_param('reg_alpha', alpha_search)
        # self.__search_one_param('reg_lambda', lambda_search)
        
        # search best value of learning_rate
        # search best value of learning_rate
        lr_search = self.config['learning_rate']
        self.__search_one_param('learning_rate', lr_search)
        
        # load model
        self.__load_model()

    def __search_one_param(self, param_name, param_value, param=None):
        if param == None:param = self.best_param.copy()
        best_param = self.best_param[param_name]
        for p_value in param_value:
            param[param_name] = p_value
            XGB_model = XGBRegressor(**param)
            XGB_model.fit(self.X_train, self.y_train, eval_set=self.eval_set, verbose=self.verbose)
            y_pre = XGB_model.predict(self.X_test)
            loss = mean_absolute_error(self.y_test, y_pre)
            if loss < self.min_loss:
                best_param = p_value
                self.min_loss = loss
                if self.verbose:print(param,'min_loss:'+str(loss),file=sys.stderr)
        self.best_param[param_name] = best_param
    
    def __search_pair_param(self, param_name1, param_value1, param_name2, param_value2, param=None):
        if param == None:param = self.best_param.copy()
        best_param1, best_param2 = self.best_param[param_name1], self.best_param[param_name2]
        for p_value1 in param_value1:
            for p_value2 in param_value2:
                param[param_name1], param[param_name2] = p_value1, p_value2 
                XGB_model = XGBRegressor(**param)
                XGB_model.fit(self.X_train, self.y_train, eval_set=self.eval_set, verbose=self.verbose)
                y_pre = XGB_model.predict(self.X_test)
                loss = mean_absolute_error(self.y_test, y_pre)
                if loss < self.min_loss:
                    best_param1, best_param2 = p_value1, p_value2
                    self.min_loss = loss
                    if self.verbose:print(param,'min_loss:'+str(loss),file=sys.stderr)
        self.best_param[param_name1], self.best_param[param_name2] = best_param1, best_param2


    def bulid_model(self, best_param=None):
        self.__load_model(best_param)
    
    def __load_model(self, best_param=None):
        if best_param == None:
            self.model = XGBRegressor(**self.best_param)
        else:
            self.model = XGBRegressor(**best_param)
        self.model.fit(self.X_train, self.y_train, eval_set=self.eval_set, verbose=self.verbose)
    
    def predict(self, X):
        return self.model.predict(np.array(X))

def train(args):
    my_data = dataset.DataSet(args.input_file, args.mode)
    X, y = my_data.inputdata, my_data.y_data
    min_loss = 1000000
    best_param, best_random_seed = None, None
    save_epoch = args.train_epochs // 10
    print("train with feature:{}".format(my_data.features))
    for random_seed in range(args.train_epochs):
        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.split_dataset_percentage, random_state=random_seed)
        XGB_model = XGB_Tree(X_train, y_train, X_test, y_test, verbose=False) 
        XGB_model.train()
        y_pre = XGB_model.model.predict(X_test)
        mae, r2, mse, pod_3 = mean_absolute_error(y_test, y_pre), r2_score(y_test, y_pre), mean_squared_error(y_test, y_pre), POD(y_test, y_pre)
        pod = POD(y_test, y_pre, mae)
        print("epochs [{}]:time_cost: {}s, best_param:{}, mean_absolute_error: {}, r2_score: {}, mean_squared_error: {}, POD_3: {}, POD: {} ".format(random_seed, 
                int(time.time()-start), XGB_model.best_param, mae, r2, mse, pod_3, pod, file=sys.stderr))
        if XGB_model.min_loss < min_loss:
           best_param, best_random_seed = XGB_model.best_param, random_seed
        if False and ((random_seed+1) % save_epoch == 0):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.split_dataset_percentage, random_state=best_random_seed)
            XGB_model = XGB_Tree(X_train, y_train, X_test, y_test)
            XGB_model.bulid_model(best_param)
            with open(args.model_save_path+'_'+str(random_seed+1),'wb') as f:
                pickle.dump(XGB_model.model, f)
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./data/cme_final_soho.json')
    parser.add_argument('--mode', type=str, default='Wind After')
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--split_dataset_percentage', type=float, default=0.2, help='The value range of split_dataset_percentage: [0,1]')
    parser.add_argument('--model_save_path', type=str, default='./model/XGB_model')
    parser.add_argument('--save_model', type=bool, default=True)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
    