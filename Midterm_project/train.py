import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

import xgboost as xgb

import bentoml

df = pd.read_csv("adult.data", sep=",", header=None)

columns = ['age', "workclass", "fnlwgt", "education", "education-num", "marital-status", 
"occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
 "native-country", "greater-than-50K"]

df.columns = columns
df.columns = df.columns.str.lower().str.replace("-", "_").str.strip()

strings = list(df.dtypes[df.dtypes == 'object'].index)

# making all the entries lower-case and seperated by undescore
for col in strings:
    df[col] = df[col].str.lower().str.replace("-", "_").str.strip()


df['us_native'] = (df['native_country'] == 'united_states').astype(int)

#converting 'us_native' back to strings
def trans_func(x):
    if x == 'united_states':
        return 'united_states'
    else:
        return 'other'

df['us_native'] = df['native_country'].apply(trans_func)

# mcahnge the target from strings to 1 and 0
df.greater_than_50k = (df.greater_than_50k == '>50k').astype(int)

# splitting the data into sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

#resettling index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# getting the targets
y_train = df_train.greater_than_50k.values
y_val = df_val.greater_than_50k.values
y_test = df_test.greater_than_50k.values

# deleting targets from data
del df_train['greater_than_50k']
del df_val['greater_than_50k']
del df_test['greater_than_50k']

numerical = ['age',
 'fnlwgt',
 'education_num',
 'capital_gain',
 'capital_loss',
 'hours_per_week']

categorical = ['workclass',
 'education',
 'marital_status',
 'occupation',
 'relationship',
 'race',
 'sex',
 'native_country',
 'us_native']

# setting up the full training set
df_full_train = df_full_train.reset_index(drop=True)

#getting the targets
y_full_train = df_full_train.greater_than_50k.values

# removing targets from training data
del df_full_train['greater_than_50k']

# converting training data to format suitable for model
dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train)

dtest = xgb.DMatrix(X_test)

# defing xgb_params and training model
xgb_params = {
    'eta': 0.3, 
    'max_depth': 4,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}


model = xgb.train(xgb_params, dfulltrain, num_boost_round=90)

# getting predictions
y_pred = model.predict(dtest)

auc = roc_auc_score(y_test, y_pred)

print("your final model's performance is %.3f" % (auc))

# saving the model with bentoml
bentoml.xgboost.save_model(
    'over_50k_model',
    model,
    custom_objects={
        'dictVectorizer': dv
    })







