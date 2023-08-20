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
Time    :   2023/01/30 19:25:37
Author  :   YangYun Group
Version :   1.0
Contact :   yangyun1073@163.com
"""
feature_config = {
    ### 'N_features':18,
    'CME_features': {'CME Average Speed': 'Speed',
            # 'CME Acceleration', # It is secondary of average and final speed
            'CME Acceleration': 'Acceleration',
            'CME Final Speed':'Speed_final',
            'CME Speed at 20 Rs': 'Speed_20',
            # 'CME Speed at 20 Rs', # It is usually not accurate
            'CME Angular Width': 'Width',
            'CME Mass': 'Mass',
            'CME Position Angle': 'PA',
            #'CME Source Region Latitude': 'Lat',
            #'CME Source Region Longitude': 'Lon'
            },

    'Solar_features': {'Solar Wind Bz': 'Bz',
            'Solar Wind Bx': 'Bx',
            'Solar Wind By': 'By',
            'Solar Wind Latitude': 'Lat',
            'Solar Wind Longitude': 'Lon',
            'Solar Wind Density': 'Rho',
            'Solar Wind Temperature': 'T',
            'Solar Wind Speed': 'V',
            'Solar Wind He Proton Ratio': 'Ratio',
            'Solar Wind Plasma Beta': 'Beta',
            'Solar Wind Pressure': 'P'
            },

    'inputdata':'cme_final_soho.json'
}


SVM_config = {'param_rbf':{"kernel":["rbf"], 
                    "C":[ 0.1,0.5, 1, 10,15,20,30,100],
                    "gamma":[1, 0.1, 0.005, 0.03, 0.001, 0.01]},

            'param_poly':{"kernel":["poly"],
                    "C": [ 0.1,0.5, 1, 10,15,20,30,100], 
                    "gamma":[1, 0.1, 0.005,0.03, 0.001,0.01],
                    "degree":[3,5,10],
                    "coef0":[0,0.1,1]},        
            
            'param_sigmoid': {"kernel":["sigmoid"],
                    "C": [ 0.1,0.5, 1, 10,15,20,30,100], 
                    "gamma":[1, 0.1, 0.005,0.03, 0.001,0.01],
                    "degree":[3,5,10]}
}


XGB_Tree_config = {
        'default_param': {'learning_rate': 0.1, 'n_estimators': 140, 'max_depth': 3, 
                        'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8, 
                        'gamma': 0,  'objective':'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1,
                         'early_stopping_rounds':150, 'nthread':1},
        'n_estimators': [5, 10, 25, 50, 75, 100,200], # [100,200,300,400,500,600,700,800,1000]
        'max_depth': range(2,8,1), # [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        'min_child_weight': range(1,8,1), # [1, 2, 3, 4, 5, 6,7, 8]
        'gamma': [0.0001, 0.001, 0.1, 0.5, 1, 10],
        'subsample': [0.01, 0.05, 0.1, 0.2, 0.5, 0.9],  
        'colsample_bytree': [0.01, 0.05, 0.1,0.2,0.5, 0.9],
        'reg_alpha': [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1], 
        'reg_lambda': [0, 0.1, 0.5],
        'learning_rate': [0.001,0.005, 0.01, 0.05, 0.3, 1]
}

Ada_Tree_config = {'default_param': {
                                'tree':{'max_depth': 5,'max_features': 5, 'min_samples_split':2},
                                'regressor': {'learning_rate': 0.1, 'n_estimators': 500}
                        },
        'n_estimators': [100,200,300,400,500,600,700,800,1000],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'max_features': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'min_samples_split': [1, 2, 3, 5, 10],
        'learning_rate':[0.001, 0.01, 0.05, 0.1, 0.3, 1]
}