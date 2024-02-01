# -*- coding: utf-8 -*-
"""baseline_notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ch3kCEGRDxHZdirIGzsPXB-FBiSvTNvT
"""

import sys
# ====================================================
# Library
# ====================================================
import os
import gc
import warnings
warnings.filterwarnings('ignore')
import random
import scipy as sp
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import joblib
import pickle
import itertools
from tqdm.auto import tqdm

# import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef, f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier

# ====================================================
# Configurations
# ====================================================
class CFG:
    VER = 1
    AUTHOR = 'Yuta.K'
    COMPETITION = 'FUDA2'
    METHOD_LIST = ['lightgbm', 'xgboost', 'catboost']
    seed = 42
    n_folds = 7
    target_col = 'MIS_Status'
    metric = 'f1_score'
    metric_maximize_flag = True
    num_boost_round = 500
    early_stopping_round = 200
    verbose = 25
    classification_lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'seed': seed,
    }
    classification_xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'random_state': seed,
    }

    classification_cat_params = {
        'learning_rate': 0.05,
        'iterations': num_boost_round,
        'random_seed': seed,
    }
    model_weight_dict = {'lightgbm': 0.50, 'xgboost': 0.10, 'catboost': 0.40}

# ====================================================
# Seed everything
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CFG.seed)

# ====================================================
# Metric
# ====================================================
# f1_score

# ====================================================
# LightGBM Metric
# ====================================================
def lgb_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'f1score', f1_score(y_true, np.where(y_pred >= 0.5, 1, 0), average='macro'), CFG.metric_maximize_flag

# ====================================================
# XGBoost Metric
# ====================================================
def xgb_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'f1score', f1_score(y_true, np.where(y_pred >= 0.5, 1, 0), average='macro')

train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

default_numerical_features = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'DisbursementGross', 'GrAppv', 'SBA_Appv', 'ApprovalFY']
default_categorical_features = ['NewExist', 'FranchiseCode', 'RevLineCr', 'LowDoc', 'UrbanRural', 'State', 'BankState', 'City', 'Sector']
add_numerical_features = ['FranchiseCode_count_encoding', 'RevLineCr_count_encoding', 'LowDoc_count_encoding', 'UrbanRural_count_encoding', 'State_count_encoding', 'BankState_count_encoding', 'City_count_encoding', 'Sector_count_encoding']
numerical_features = add_numerical_features + default_numerical_features
categorical_features = ['RevLineCr', 'LowDoc', 'UrbanRural', 'State', 'Sector']
features = numerical_features + categorical_features

def Preprocessing(input_df: pd.DataFrame()) -> pd.DataFrame():
    def deal_missing(input_df: pd.DataFrame()) -> pd.DataFrame():
        output_df = input_df.copy()
        for col in ['RevLineCr', 'LowDoc', 'BankState', 'DisbursementDate']:
            output_df[col] = input_df[col].fillna('[UNK]')
        return output_df
    def clean_money(input_df: pd.DataFrame()) -> pd.DataFrame():
        output_df = input_df.copy()
        for col in ['DisbursementGross', 'GrAppv', 'SBA_Appv']:
            output_df[col] = input_df[col].str[1:].str.replace(',', '').str.replace(' ', '').astype(float)
        return output_df
    output_df = deal_missing(input_df)
    output_df = clean_money(output_df)
    output_df['NewExist'] = np.where(input_df['NewExist'] == 1, 1, 0)
    def make_features(input_df: pd.DataFrame()) -> pd.DataFrame():
        output_df = input_df.copy()
        # いろいろ特徴量作成を追加する
        return output_df
    output_df = make_features(output_df)
    return output_df

train_df = Preprocessing(train_df)
test_df = Preprocessing(test_df)

"""（以下はPreprocessingに本来組み込むべきだが，コードが煩雑になるので，いったん切り出している．）"""

for col in ['FranchiseCode', 'RevLineCr', 'LowDoc', 'UrbanRural', 'State', 'BankState', 'City', 'Sector']:
    count_dict = dict(train_df[col].value_counts())
    train_df[f'{col}_count_encoding'] = train_df[col].map(count_dict)
    test_df[f'{col}_count_encoding'] = test_df[col].map(count_dict).fillna(1).astype(int)

for col in categorical_features:
    encoder = LabelEncoder()
    encoder.fit(train_df[col])
    train_df[col] = encoder.transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

def lightgbm_training(x_train: pd.DataFrame, y_train: pd.DataFrame, x_valid: pd.DataFrame, y_valid: pd.DataFrame, features: list, categorical_features: list):
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(x_valid, y_valid, categorical_feature=categorical_features)
    model = lgb.train(
                params = CFG.classification_lgb_params,
                train_set = lgb_train,
                num_boost_round = CFG.num_boost_round,
                valid_sets = [lgb_train, lgb_valid],
                feval = lgb_metric,
                callbacks=[lgb.early_stopping(stopping_rounds=CFG.early_stopping_round,
                                              verbose=CFG.verbose)]
            )
    # Predict validation
    valid_pred = model.predict(x_valid)
    return model, valid_pred
def xgboost_training(x_train: pd.DataFrame, y_train: pd.DataFrame, x_valid: pd.DataFrame, y_valid: pd.DataFrame, features: list, categorical_features: list):
    xgb_train = xgb.DMatrix(data=x_train, label=y_train)
    xgb_valid = xgb.DMatrix(data=x_valid, label=y_valid)
    model = xgb.train(
                CFG.classification_xgb_params,
                dtrain = xgb_train,
                num_boost_round = CFG.num_boost_round,
                evals = [(xgb_train, 'train'), (xgb_valid, 'eval')],
                early_stopping_rounds = CFG.early_stopping_round,
                verbose_eval = CFG.verbose,
                feval = xgb_metric,
                maximize = CFG.metric_maximize_flag,
            )
    # Predict validation
    valid_pred = model.predict(xgb.DMatrix(x_valid))
    return model, valid_pred
def catboost_training(x_train: pd.DataFrame, y_train: pd.DataFrame, x_valid: pd.DataFrame, y_valid: pd.DataFrame, features: list, categorical_features: list):
    cat_train = Pool(data=x_train, label=y_train, cat_features=categorical_features)
    cat_valid = Pool(data=x_valid, label=y_valid, cat_features=categorical_features)
    model = CatBoostClassifier(**CFG.classification_cat_params)
    model.fit(cat_train,
              eval_set = [cat_valid],
              early_stopping_rounds = CFG.early_stopping_round,
              verbose = CFG.verbose,
              use_best_model = True)
    # Predict validation
    valid_pred = model.predict_proba(x_valid)[:, 1]
    return model, valid_pred

def gradient_boosting_model_cv_training(method: str, train_df: pd.DataFrame, features: list, categorical_features: list):
    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(train_df))
    oof_fold = np.zeros(len(train_df))
    kfold = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    for fold, (train_index, valid_index) in enumerate(kfold.split(train_df)):
        print('-'*50)
        print(f'{method} training fold {fold+1}')

        x_train = train_df[features].iloc[train_index]
        y_train = train_df[CFG.target_col].iloc[train_index]
        x_valid = train_df[features].iloc[valid_index]
        y_valid = train_df[CFG.target_col].iloc[valid_index]
        if method == 'lightgbm':
            model, valid_pred = lightgbm_training(x_train, y_train, x_valid, y_valid, features, categorical_features)
        if method == 'xgboost':
            model, valid_pred = xgboost_training(x_train, y_train, x_valid, y_valid, features, categorical_features)
        if method == 'catboost':
            model, valid_pred = catboost_training(x_train, y_train, x_valid, y_valid, features, categorical_features)

        # Save best model
        pickle.dump(model, open(f'{method}_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl', 'wb'))
        # Add to out of folds array
        oof_predictions[valid_index] = valid_pred
        oof_fold[valid_index] = fold + 1
        del x_train, x_valid, y_train, y_valid, model, valid_pred
        gc.collect()

    # Compute out of folds metric
    score = f1_score(train_df[CFG.target_col], oof_predictions >= 0.5, average='macro')
    print(f'{method} our out of folds CV f1score is {score}')
    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame({CFG.target_col: train_df[CFG.target_col], f'{method}_prediction': oof_predictions, 'fold': oof_fold})
    oof_df.to_csv(f'oof_{method}_seed{CFG.seed}_ver{CFG.VER}.csv', index = False)

def Learning(input_df: pd.DataFrame, features: list, categorical_features: list):
    for method in CFG.METHOD_LIST:
        gradient_boosting_model_cv_training(method, input_df, features, categorical_features)

Learning(train_df, features, categorical_features)

def lightgbm_inference(x_test: pd.DataFrame):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(open(f'lightgbm_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl', 'rb'))
        # Predict
        pred = model.predict(x_test)
        test_pred += pred
    return test_pred / CFG.n_folds
def xgboost_inference(x_test: pd.DataFrame):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(open(f'xgboost_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl', 'rb'))
        # Predict
        pred = model.predict(xgb.DMatrix(x_test))
        test_pred += pred
    return test_pred / CFG.n_folds

def catboost_inference(x_test: pd.DataFrame):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(open(f'catboost_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl', 'rb'))
        # Predict
        pred = model.predict_proba(x_test)[:, 1]
        test_pred += pred
    return test_pred / CFG.n_folds

def gradient_boosting_model_inference(method: str, test_df: pd.DataFrame, features: list, categorical_features: list):
    x_test = test_df[features]
    if method == 'lightgbm':
        test_pred = lightgbm_inference(x_test)
    if method == 'xgboost':
        test_pred = xgboost_inference(x_test)
    if method == 'catboost':
        test_pred = catboost_inference(x_test)
    return test_pred

def Predicting(input_df: pd.DataFrame, features: list, categorical_features: list):
    output_df = input_df.copy()
    output_df['pred_prob'] = 0
    for method in CFG.METHOD_LIST:
        output_df[f'{method}_pred_prob'] = gradient_boosting_model_inference(method, input_df, features, categorical_features)
        output_df['pred_prob'] += CFG.model_weight_dict[method] * output_df[f'{method}_pred_prob']
    return output_df

test_df = Predicting(test_df, features, categorical_features)

def Postprocessing(train_df: pd.DataFrame(), test_df: pd.DataFrame()) -> (pd.DataFrame(), pd.DataFrame()):
    train_df['pred_prob'] = 0
    for method in CFG.METHOD_LIST:
        oof_df = pd.read_csv(f'oof_{method}_seed{CFG.seed}_ver{CFG.VER}.csv')
        train_df['pred_prob'] += CFG.model_weight_dict[method] * oof_df[f'{method}_prediction']
    best_score = 0
    best_v = 0
    for v in tqdm(np.arange(1000) / 1000):
        score = f1_score(oof_df[CFG.target_col], train_df[f'pred_prob'] >= v, average='macro')
        if score > best_score:
            best_score = score
            best_v = v
    print(best_score, best_v)
    test_df['target'] = np.where(test_df['pred_prob'] >= best_v, 1, 0)
    return train_df, test_df

train_df, test_df = Postprocessing(train_df, test_df)

test_df[['target']].to_csv(f'seed{CFG.seed}_ver{CFG.VER}_{CFG.AUTHOR}_submission.csv', header=False)

"""特徴量の重要度を確認する方法"""

model = pickle.load(open(f'lightgbm_fold1_seed42_ver1.pkl', 'rb'))
importance_df = pd.DataFrame(model.feature_importance(), index=features, columns=['importance'])
importance_df['importance'] = importance_df['importance'] / np.sum(importance_df['importance'])
importance_df.sort_values('importance', ascending=False)
