from app import WindData
from app import Model
from app import FeatureSet

from app import Knn
from app import Mcp
from app import LinReg
from app import Svm
from app import DecTree
from app import RandForest
from app import NeuralNetwork
from app import LstmNetwork

import numpy as np


if __name__ == '__main__':

    # ----------------------------------------------------------------- #
    #                             SETTINGS                              #
    # ----------------------------------------------------------------- #

    do_preprocessing = False
    do_model_preparation = False
    do_performance = False
    do_comparison = True

    # ------------------------ FEATURE VARIANTS ----------------------- #

    feat_variants = {1: FeatureSet(1, ['WS-92']),
                     2: FeatureSet(2, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500']),
                     3: FeatureSet(3, ['WS-92', 'WD-92']),
                     4: FeatureSet(4, ['WS-92', 'WD_SIN', 'WD_COS']),
                     5: FeatureSet(5, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD_SIN',
                                       'WD_COS']),
                     6: FeatureSet(6, ['WS-92', 'WD_SIN', 'WD_COS', 'RMOL3']),
                     7: FeatureSet(7, ['WS-92', 'WD_SIN', 'WD_COS', 'RMOL5']),
                     8: FeatureSet(8, ['WS-92', 'WD_SIN', 'WD_COS', 'season']),
                     9: FeatureSet(9, ['WS-92', 'WD_seaside']),
                     10: FeatureSet(10, ['WS-92', 'WD_sectors']),
                     11: FeatureSet(11, ['WS-92', 'T_diff']),
                     12: FeatureSet(12, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'T_diff']),
                     13: FeatureSet(13, ['WS-75', 'WS-100', 'T_diff']),
                     14: FeatureSet(14, ['WS-75', 'WS-100']),
                     15: FeatureSet(15, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD-92']),
                     16: FeatureSet(16, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD-92',
                                         'T_diff']),
                     17: FeatureSet(17, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD-92',
                                         'T_diff', 'season']),
                     18: FeatureSet(18, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD-92',
                                         'T_diff', 'RMOL3']),
                     19: FeatureSet(19, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD-92',
                                         'Timestamp']),
                     20: FeatureSet(20, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD-92',
                                         'daytime']),
                     21: FeatureSet(21, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD-92',
                                         'Timestamp', 'daytime', 'T_diff']),
                     22: FeatureSet(22, ['WS-50', 'WS-75', 'WS-100', 'WS-150', 'WS-200', 'WS-250', 'WS-500', 'WD-92',
                                         'Timestamp', 'daytime', 'T_diff', 'season'])
                     }

    # ---------------------- DEFINE PERFORMANCES ---------------------- #

    performances = {}

    for i in range(1, 23, 1):
        performances[f'mcp{i}'] = {'Model Type': 'mcp', 'Feature Set': i}
        performances[f'T-{i}'] = {'Model Type': 'dtree', 'Feature Set': i, 'max_depth': None,
                                  'md_selection': np.arange(1, 10, 1), 'criterion': 'mse', 'random_state': 0}
        performances[f'F-{i}'] = {'Model Type': 'rFor', 'Feature Set': i, 'max_depth': None, 'n_estimators': None,
                                  'md_selection': np.arange(4, 6, 1), 'nest_selection': np.arange(1, 50, 1)}
        performances[f'F_1000-{i}'] = {'Model Type': 'rFor', 'Feature Set': i, 'max_depth': None, 'n_estimators': 1000,
                                       'md_selection': np.arange(4, 6, 1), 'nest_selection': None}
        performances[f'knn_uni-{i}'] = {'Model Type': 'knn', 'Feature Set': i, 'k_selection': np.arange(156, 160, 2),
                                        'weights': 'uniform'}
        performances[f'knn_dist-{i}'] = {'Model Type': 'knn', 'Feature Set': i, 'k_selection': np.arange(156, 160, 2),
                                         'weights': 'distance'}
        performances[f'linR-{i}'] = {'Model Type': 'linR', 'Feature Set': i}
        performances[f'svm_rbf-{i}'] = {'Model Type': 'svm', 'Feature Set': i, 'kernel': 'rbf'}
        performances[f'svm_poly-{i}'] = {'Model Type': 'svm', 'Feature Set': i, 'kernel': 'poly'}
        performances[f'svm_lin-{i}'] = {'Model Type': 'svm', 'Feature Set': i, 'kernel': 'linear'}
        performances[f'nn-{i}'] = {'Model Type': 'nn', 'Feature Set': i, 'n_iter': 10000, 'learning_rate': 0.001}
        performances[f'lstm-{i}'] = {'Model Type': 'lstm', 'Feature Set': i, 'seq_len': 6, 'hidden_dim': 24,
                                     'learning_rate': 0.001, 'n_epochs': 1000, 'batch_size': 128, 'weight_decay': 0.0}

    # ----------------------------------------------------------------- #
    #                               START                               #
    # ----------------------------------------------------------------- #

    # Creating WindData object
    ij = WindData('Ijmuiden')

    # ----------------------------------------------------------------- #
    #                           PREPROCESSING                           #
    # ----------------------------------------------------------------- #

    if do_preprocessing:
        ij.preprocessing()
        print('Preprocessing done')

    # ----------------------------------------------------------------- #
    #                         MODEL PREPARATION                         #
    # ----------------------------------------------------------------- #

    # Target (Measurements) will not be normalized or encoded
    if do_model_preparation:
        # Split data, save split data
        ij.model_preparation(save_measurements=True)
        # Normalize data, split data, save split data
        ij.model_preparation(norm=True)
        # Do one hot encoding for categories, split data, save split data
        ij.model_preparation(do_ohe=True)
        # Do one hot encoding for categories, normalize data, split data, save split data
        ij.model_preparation(do_ohe=True, norm=True)
        # Do label encoding for categories, split data, save split data
        ij.model_preparation(do_le=True)
        # Do label encoding for categories, normalize data, split data, save split data
        ij.model_preparation(do_le=True, norm=True)

        print('Model Preparation done')

    # ----------------------------------------------------------------- #
    #                               MODELS                              #
    # ----------------------------------------------------------------- #

    # Creating Model object
    ij.simulations['df'] = ij.reload_data(filename='clean_newa')
    ij.measurements['df'] = ij.reload_data(filename='clean_measurements')
    ijm = Model(ij.name, ij.simulations['df'], ij.measurements['df'])

    if do_performance:
        for ip, prf in performances.items():
            feat_variants[prf['Feature Set']].save_feature_set(settings_path=ijm.get_settings_path(feat_settings=True))

            # ------------------------------ MCP ------------------------------ #

            if prf['Model Type'] == 'mcp':
                ij.load_split_data(feature_set=FeatureSet(0, ['WS-92']).feature_set, norm=False)
                model_features = {k: v for k, v in ij.simulations.items() if k in ['Train', 'Val', 'Test']}
                model_targets = {k: v for k, v in ij.measurements.items() if k in ['Train', 'Val', 'Test']}

                ij_mcp = Mcp(ij.name, model_features, model_targets)
                ij_mcp.perform_mcp(model_name=ip)

                print(f'MCP {ip} done')

            # ------------------------------ KNN ------------------------------ #

            if prf['Model Type'] == 'knn':
                ij.load_split_data(feature_set=feat_variants[prf['Feature Set']].feature_set, norm=True, ohe=True)
                model_features = {k: v for k, v in ij.simulations.items() if k in ['Train', 'Val', 'Test']}
                model_targets = {k: v for k, v in ij.measurements.items() if k in ['Train', 'Val', 'Test']}
                ij_knn = Knn(ij.name, model_features, model_targets)
                ij_knn.perform_knn(model_name=ip, features=feat_variants[prf['Feature Set']],
                                   k_selection=prf['k_selection'], knn_weights=prf['weights'])

                print(f'KNN {ip} done')

            # ----------------------- LINEAR REGRESSION ----------------------- #

            if prf['Model Type'] == 'linR':
                ij.load_split_data(feat_variants[prf['Feature Set']].feature_set, norm=True, ohe=True)
                model_features = {k: v for k, v in ij.simulations.items() if k in ['Train', 'Val', 'Test']}
                model_targets = {k: v for k, v in ij.measurements.items() if k in ['Train', 'Val', 'Test']}
                ij_linreg = LinReg(ij.name, model_features, model_targets)
                ij_linreg.perform_linreg(model_name=ip, features=feat_variants[prf['Feature Set']])

                print(f'LinReg {ip} done')

            # ------------------------------ SVM ------------------------------ #

            if prf['Model Type'] == 'svm':
                ij.load_split_data(feat_variants[prf['Feature Set']].feature_set, norm=True, ohe=True)     # TODO check input data
                model_features = {k: v for k, v in ij.simulations.items() if k in ['Train', 'Val', 'Test']}
                model_targets = {k: v for k, v in ij.measurements.items() if k in ['Train', 'Val', 'Test']}
                ij_svm = Svm(ij.name, model_features, model_targets)
                ij_svm.perform_svm(model_name=ip, features=feat_variants[prf['Feature Set']], kernel=prf['kernel'])

                print(f'SVM {ip} done')

            # ------------------------ DECISION TREES ------------------------- #

            if prf['Model Type'] == 'dtree':
                ij.load_split_data(feat_variants[prf['Feature Set']].feature_set, norm=False, le=True)
                model_features = {k: v for k, v in ij.simulations.items() if k in ['Train', 'Val', 'Test']}
                model_targets = {k: v for k, v in ij.measurements.items() if k in ['Train', 'Val', 'Test']}
                ij_dtree = DecTree(ij.name, model_features, model_targets)
                ij_dtree.perform_dtree(model_name=ip, features=feat_variants[prf['Feature Set']],
                                       max_depth=prf['max_depth'], md_selection=prf['md_selection'],
                                       criterion=prf['criterion'], random_state=prf['random_state'])

                print(f'Decision Tree {ip} done')

            # ------------------------- RANDOM FOREST ------------------------- #

            if prf['Model Type'] == 'rFor':
                ij.load_split_data(feat_variants[prf['Feature Set']].feature_set, norm=False, le=True)
                model_features = {k: v for k, v in ij.simulations.items() if k in ['Train', 'Val', 'Test']}
                model_targets = {k: v for k, v in ij.measurements.items() if k in ['Train', 'Val', 'Test']}
                ij_ranfor = RandForest(ij.name, model_features, model_targets)
                ij_ranfor.perform_rforest(model_name=ip, features=feat_variants[prf['Feature Set']],
                                          max_depth=prf['max_depth'], md_selection=prf['md_selection'],
                                          n_estimators=prf['n_estimators'], nest_selection=prf['nest_selection'])

                print(f'Random Forest {ip} done')

            # ------------------------ NEURAL NETWORKS ------------------------ #

            if prf['Model Type'] == 'nn':
                ij.load_split_data(feat_variants[prf['Feature Set']].feature_set, norm=True, use_torch=True)
                model_features = {k: v for k, v in ij.simulations.items() if k in ['Train', 'Val', 'Test']}
                model_targets = {k: v for k, v in ij.measurements.items() if k in ['Train', 'Val', 'Test']}
                ij_nn = NeuralNetwork(ij.name, model_features, model_targets)
                if ij.n_features > 1:
                    ij_nn.perform_nn(model_name=ip, features=feat_variants[prf['Feature Set']],
                                     d_in=ij.n_features, n_iter=prf['n_iter'],
                                     learning_rate=prf['learning_rate'])

                print(f'Neural Network {ip} done')

            # ----------------------------- LSTM ------------------------------ #

            if prf['Model Type'] == 'lstm':
                ij.load_split_data(feat_variants[prf['Feature Set']].feature_set, norm=True)
                model_features = {k: v for k, v in ij.simulations.items() if k in ['Train', 'Val', 'Test']}
                model_targets = {k: v for k, v in ij.measurements.items() if k in ['Train', 'Val', 'Test']}
                ij_lstm = LstmNetwork(ij.name, model_features, model_targets)
                ij_lstm.perform_lstm(model_name=ip, features=feat_variants[prf['Feature Set']], seq_len=prf['seq_len'],
                                     hidden_dim=prf['hidden_dim'], learning_rate=prf['learning_rate'],
                                     n_epochs=prf['n_epochs'], batch_size=prf['batch_size'],
                                     weight_decay=prf['weight_decay'])

                print(f'LSTM {ip} done')

    # ----------------------------------------------------------------- #
    #                            COMPARISON                             #
    # ----------------------------------------------------------------- #

    if do_comparison:
        ijm.compare_models()
        ijm.rank_performances()
