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
Time    :   2023/01/30 19:26:20
Author  :   YangYun Group
Version :   1.0
Contact :   yangyun1073@163.com
"""
import dataset
import numpy as np
import pickle
import config
import json
import sys
import argparse
import sklearn.metrics 

def POD(y_data, y_pre, M=3):
    abs_error = np.abs(y_pre - y_data)
    return round(sum(abs_error<M)/len(y_data),4)

def mean_absolute_error(y_test, y_pre):
    try:
        loss = sklearn.metrics.mean_absolute_error(y_test, y_pre)
    except:
        loss = sklearn.metrics.mean_absolute_error(y_test, np.zeros(y_test.shape))
    return loss

def predict(model_path, input_file, mode='Wind After'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    X_data = dataset.X_data_process(input_file, mode)
    return model.predict(X_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./data/cme_final_soho_valid.json')
    parser.add_argument('--mode', type=str, default='Wind After')
    parser.add_argument('--model_path', type=str, default='./model/XGB_model')
    args = parser.parse_args()

    print(predict(args.model_path, args.input_file, args.mode))

if __name__ == '__main__':
    main()