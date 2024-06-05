import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('city_day.csv')
dataset = dataset.drop(columns=['City','Date'])
dataset.isna().sum()/dataset.shape[0]
dataset = dataset[dataset['AQI'].isna()==False]
dataset.isna().sum()/dataset.shape[0]
columns_to_remove_null_rows = [i for i in dataset.columns if dataset[i].isna().sum()/dataset.shape[0]<=0.1 and dataset[i].isna().sum()>0]

columns_to_impute = [i for i in dataset.columns if i not in columns_to_remove_null_rows and dataset[i].isna().sum()>0]

print("Columns in which rows will be removed:", columns_to_remove_null_rows)
print("Columns which will be imputed:", columns_to_impute)

from sklearn.impute import SimpleImputer

def remove_rows(data, column):
  return data[data[column].isna()==False]

for i in columns_to_remove_null_rows:
  dataset = remove_rows(dataset, i)

imputer = SimpleImputer()
dataset[columns_to_impute] = imputer.fit_transform(dataset[columns_to_impute])
dataset.isna().sum()/dataset.shape[0]
dataset[[i for i in dataset.columns if i not in ["AQI_Bucket","AQI"]]].hist(bins=30, figsize=(15, 10))
plt.show()
from sklearn.preprocessing import StandardScaler

dataset_norm_test = dataset[[i for i in dataset.columns if i not in ["AQI_Bucket","AQI"]]].copy()
scaler = StandardScaler()
dataset_norm_test = pd.DataFrame(scaler.fit_transform(dataset_norm_test),columns=dataset_norm_test.columns)
dataset_norm_test.hist(bins=30, figsize=(15, 10))
plt.show()
import scipy.stats as stats

dataset_yeo_test = dataset[[i for i in dataset.columns if i not in ["AQI_Bucket","AQI"]]].copy()

for i in dataset_yeo_test.columns:
  yeo_t,param = stats.yeojohnson(dataset_yeo_test[i])
  plt.hist(yeo_t,bins=25)
  plt.xlabel(i)
  plt.show()
columns_to_binarize = ['Benzene','Toluene','Xylene']
columns_to_yeo = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3']

print('Columns to binarize:',columns_to_binarize)
print('Columns to yeo transform: ', columns_to_yeo)
dataset[columns_to_binarize].hist(bins=60, figsize=(15, 10))
plt.show()
binarize_threshold = {
    'Benzene' : 30,
    'Xylene' : 30,
    'Toluene' : 30,
}

def binarize(value, thresold):
  if value<=thresold:
    return 0
  return 1

for col in columns_to_binarize:
  new_col = col+"_binarized"
  dataset[new_col] = dataset[col].apply(lambda x: binarize(x, binarize_threshold[col]))
yeo_transform_params = {}

for col in columns_to_yeo:
  yeo_t,param = stats.yeojohnson(dataset_yeo_test[col])
  yeo_transform_params[col] = param
  dataset[col] = yeo_t
yeo_transform_params
import seaborn as sns

dataset_corr = dataset[[i for i in dataset.columns if i!="AQI_Bucket"]]

plt.figure(figsize=(12,10))
cor = dataset_corr.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
dataset.drop(columns = columns_to_binarize+['NOx'],inplace=True)
dataset['AQI'] = np.log(dataset['AQI'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset[[i for i in dataset.columns if i not in ["AQI","AQI_Bucket"]]],
    dataset["AQI"],
    test_size=0.2,
    random_state=100
    )


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)

X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

from sklearn.linear_model import LinearRegression


model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score

pred = model_lr.predict(X_train)

print('train mse: {}'.format(
    mean_squared_error((y_train), (pred))))
print('train rmse: {}'.format(
    mean_squared_error((y_train), (pred), squared=False)))
print('train r2: {}'.format(
    r2_score((y_train), (pred))))
print()

# make predictions for test set
pred = model_lr.predict(X_test)

# determine mse, rmse and r2
print('test mse: {}'.format(
    mean_squared_error((y_test), (pred))))
print('test rmse: {}'.format(
    mean_squared_error((y_test), (pred), squared=False)))
print('test r2: {}'.format(
    r2_score((y_test), (pred))))
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score

pred = model.predict(X_train)

print('train mse: {}'.format(
    mean_squared_error((y_train), (pred))))
print('train rmse: {}'.format(
    mean_squared_error((y_train), (pred), squared=False)))
print('train r2: {}'.format(
    r2_score((y_train), (pred))))
print()

# make predictions for test set
pred = model.predict(X_test)

# determine mse, rmse and r2
print('test mse: {}'.format(
    mean_squared_error((y_test), (pred))))
print('test rmse: {}'.format(
    mean_squared_error((y_test), (pred), squared=False)))
print('test r2: {}'.format(
    r2_score((y_test), (pred))))
from sklearn.linear_model import Ridge

model = Ridge(alpha = 1)
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score

pred = model.predict(X_train)

print('train mse: {}'.format(
    mean_squared_error((y_train), (pred))))
print('train rmse: {}'.format(
    mean_squared_error((y_train), (pred), squared=False)))
print('train r2: {}'.format(
    r2_score((y_train), (pred))))
print()

# make predictions for test set
pred = model.predict(X_test)

# determine mse, rmse and r2
print('test mse: {}'.format(
    mean_squared_error((y_test), (pred))))
print('test rmse: {}'.format(
    mean_squared_error((y_test), (pred), squared=False)))
print('test r2: {}'.format(
    r2_score((y_test), (pred))))
import joblib

joblib_file = "joblib_RL_Model.pkl"
joblib.dump(model_lr, joblib_file)
joblib_file_scaler = "scaler.pkl"
joblib.dump(scaler, joblib_file_scaler)