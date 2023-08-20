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
Time    :   2023/01/30 19:25:58
Author  :   YangYun Group
Version :   1.0
Contact :   yangyun1073@163.com
"""

import numpy as np
import config as config 
import sys
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import random
import pickle




def time_convert(x):
    return  datetime.strptime(str(x), '%Y-%m-%dT%H:%M:%S')

class NormData:
    """ NormData """
    def __init__(self,x):
        self.x = np.array(x)
        self.__norm()
        
    def __norm(self):
        x = self.x.copy()
        shape_ = (1,-1) if len(x.shape)==2 else (-1)
        mean_x = np.mean(x, axis=0).reshape(shape_)
        std_x = np.std(x, axis=0).reshape(shape_)
        self.data = (x-mean_x)/std_x
        self._std_x = std_x
        self._mean_x = mean_x
    
    def norm(self,x):
        return (x-self._mean_x)/self._std_x

    def renorm(self, x):
        return x*self._std_x + self._mean_x

def split_dataset(input_file='./data/cme_final_soho.json', split_dataset_percentage=0.2, random_seed=2):
    """ Help split data to trian dataset and valid dataset """
    random.seed(random_seed)
    with open(input_file) as f:
        info = json.loads(f.read())
    nums = len(info['CME'])
    valid_index = random.sample(range(nums), int(nums*split_dataset_percentage))
    valid_index.sort()
    train_index = [i for i in range(nums) if i not in valid_index]

    def _split(dict_):
        res_valid, res_train = dict(), dict()
        for key in dict_:
            value = dict_[key]
            if isinstance(value, dict):
                res_valid[key], res_train[key] = _split(value)
            else:
                res_valid[key], res_train[key] = [value[i] for i in valid_index], [value[i] for i in train_index]
        return res_valid, res_train
    valid_info, train_info = _split(info)
    valid_info, train_info = json.dumps(valid_info), json.dumps(train_info)
    with open(input_file.rsplit('.',1)[0] + '_valid.json', 'w+') as f:
        f.write(valid_info)
    with open(input_file.rsplit('.',1)[0] + '_train.json', 'w+') as f:
        f.write(train_info)

def X_data_process(input_data, mode = 'Wind After', transfer_path='./data/transfer.obj'):
    """ process X_data with norm transfer"""
    with open(input_data) as f:
        info = json.loads(f.read())
    nums = len(info['CME'])
    inputdata =[]
    CME_features = config.feature_config['CME_features']
    Solar_features = config.feature_config['Solar_features']
    # Data loading
    for feature in sorted(list(CME_features.keys())):
        if CME_features[feature] in info:
            inputdata.append(np.array(info[CME_features[feature]], dtype=float))
        else:
            print('Warining:Missing CME_key {}'.foramt(feature), file=sys.stderr)
            inputdata.append(np.zeros(nums), dtype=float)
    sloar_info = info[mode]
    for feature in sorted(list(Solar_features.keys())):
        if Solar_features[feature] in sloar_info:
            inputdata.append(np.array(sloar_info[Solar_features[feature]], dtype=float))
        else:
            print('Warining:Missing Sloar_key {}'.foramt(feature), file=sys.stderr)
            inputdata.append(np.zeros(nums), dtype=float)
    # Data normalization processing
    inputdata = np.array(inputdata).transpose()
    with open(transfer_path,'rb') as f:
        scaler = pickle.load(f)
    X_data = scaler.transform(inputdata)
    return X_data

def y_data_process(input_data):
    with open(input_data) as f:
        info = json.loads(f.read())
    nums = len(info['Actual'])
    start = list(map(time_convert, info['Time']))
    stop = list(map(time_convert, info['Actual']))
    y_data = np.array([(stop[i]-start[i]).total_seconds()/3600 for i in range(nums)])
    return y_data

class DataSet:
    def __init__(self, input_file='./data/cme_final_soho_train.json', mode='Wind After'):
        """initial DataSet with config""" 
        self.params = config.feature_config
        self.CME_features = self.params['CME_features']
        self.Solar_features = self.params['Solar_features']
        self.features = []
        self.input_file = input_file
        self.mode = mode
        self.get_inputdata(self.mode)
        

    def get_inputdata(self, mode='Wind After', input_file=None):
        """get input data of json type, and load sloar info with mode in ['Wind After', 'Wind Before']""" 
        input_file = self.input_file if not input_file else input_file
        with open(input_file) as f:
            info = json.loads(f.read())
        
        self._nums = len(info['CME'])
        inputdata =[]

        # Data loading
        for feature in sorted(list(self.CME_features.keys())):
            if self.CME_features[feature] in info:
               inputdata.append(np.array(info[self.CME_features[feature]], dtype=float))
            else:
                print('Warining:Missing CME_key {}'.foramt(feature), file=sys.stderr)
                inputdata.append(np.zeros(self._nums), dtype=float)
            self.features.append(feature)
        sloar_info = info[mode]
        for feature in sorted(list(self.Solar_features.keys())):
            if self.Solar_features[feature] in sloar_info:
                inputdata.append(np.array(sloar_info[self.Solar_features[feature]], dtype=float))
            else:
                print('Warining:Missing Sloar_key {}'.foramt(feature), file=sys.stderr)
                inputdata.append(np.zeros(self._nums), dtype=float)
            self.features.append(feature)

        # Data normalization processing
        inputdata = np.array(inputdata).transpose()
        self.rawdata = inputdata.copy()
        scaler = StandardScaler()
        self.scaler = scaler
        self.inputdata = self.scaler.fit_transform(inputdata)
        with open('./data/transfer.obj','wb') as f:
            pickle.dump(scaler, f)
        start = list(map(time_convert, info['Time']))
        stop = list(map(time_convert, info['Actual']))
        self.y_data = np.array([(stop[i]-start[i]).total_seconds()/3600 for i in range(self._nums)])

if __name__ == '__main__':
    globals()[sys.argv[1]](*sys.argv[2:])






        
        
    
