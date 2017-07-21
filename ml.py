from flask import Flask, redirect, render_template, url_for, request
import requests
import simplejson as json
import numpy as np 
import math
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure,ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.embed import components 
from bokeh.models import HoverTool, Range1d, FixedTicker, FuncTickFormatter
from datetime import datetime
from sklearn import base
from sklearn import datasets, linear_model, utils, preprocessing
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn import neighbors
from sklearn import ensemble
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer


def Dict_Flattener_transform(X):
    # X will come in as a list of dicts. Return a list of dicts.
    l = []
    for x in X:
        dic = dict()
        for a in x: 
            #x: dict like {key1:value1, key2: {kkey3: value3, kkey4: value4}}
            #a: entry like key1:value1 or key2: {kkey3: value3, kkey4: value4}
            #dic[a] = 1
            value1 = x[a]
            #print type(value1)
            
            if isinstance(value1, bool):
                    if value1 == True:
                        dic[a] = int(1)
                    else:
                        dic[a] = int(0)
                    continue
                    
            if isinstance(value1, dict): 
                for b in value1:
                    value2 = value1[b]
                    #print type(value2)
                    if isinstance(value2, bool):
                        if value2 == True:
                            dic[a+'_'+b] = int(1)
                        else:
                            dic[a+'_'+b] = int(0)
                        continue
                    
                    if isinstance(value2, dict): 
                        print 'Error: more than 2 layers of dict !'
                    else:
                        if isinstance(value2, str): #type(value2)==type(''):
                            dic[a+'_'+b+'_'+value2] = 1
                        if isinstance(value2, (int, float)):
                            dic[a+'_'+b] = value2
                            
            else:
                if isinstance(value1, str): #(value1)==type(''):
                    dic[a+'_'+value1] = 1
                if isinstance(value1, (int, float)):
                    dic[a] = value1
        l.append(dic)
    return l


def get_attr_dict(attr_str, title_str, message_str):
    dic = dict()
    text = title_str + '\n'+ message_str
    
    m = re.search(r'cylinders: (\d{1,2}) cylinders[#|$]' , attr_str)
    if m :  dic['CYLINDERS'] = m.group(1)
        
    m = re.search(r'fuel: (\w+)[#|$]' , attr_str)
    if m :  dic['FUEL'] = m.group(1)
        
    m = re.search(r'paint color: (\w+)[#|$]' , attr_str)
    if m :  dic['COLOR'] = m.group(1)
        
    m = re.search(r'condition: (\w+)[#|$]' , attr_str)
    if m :  dic['CONDITION'] = m.group(1)
        
    m = re.search(r'title status: (\w+)[#|$]' , attr_str)
    if m :  dic['TITLE STATUS'] = m.group(1)
        
    m = re.search(r'transmission: (\w+)[#|$]' , attr_str)
    if m :  dic['TRANSMISSION'] = m.group(1)
        
    m = re.search(r'type: (\w+)[#|$]' , attr_str)
    if m :  dic['TYPE'] = m.group(1)
        
    m = re.search(r'size: (\w+)[#|$]' , attr_str)
    if m :  dic['SIZE'] = m.group(1)
        
    m = re.search(r'drive: (\w+)[#|$]' , attr_str)
    if m :  dic['DRIVE'] = m.group(1)
        
    return dic


def df_Dict_Transform(df1):
    l = []
    rows_df , cols_df = df1.shape
    for i in range(rows_df):
        row = df1.ix[i]
        dic = dict()
        posttime = row['POSTTIME']
        md = datetime.strptime( posttime,'%Y-%m-%d %H:%M' ) 
        dic['POSTDAY'] = md.timetuple().tm_yday
        dic['CITY'] = row['CITY']
        dic['STATE'] = row['STATE']
        dic['MAKE'] = row['MAKE']
        dic['MODEL'] = row['MODEL']
        dic['YEAR'] = int(row['YEAR'])
        dic['MILES'] = int(row['MILES'])
        dic['ATTR'] = get_attr_dict(row['ATTR'],row['TITLE'],row['MESSAGE'])
        #print dic
        l.append(dic)    
    return l


class EnsembleTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, base_estimator, residual_estimators):
        self.base_estimator = base_estimator
        self.residual_estimators = residual_estimators
    
    def fit(self, X, y):
        import numpy as np 
        self.base_estimator.fit(X, y)
        y_err = y - self.base_estimator.predict(X)
        for est in self.residual_estimators:
            est.fit(X, y_err)
        return self
    
    def transform(self, X):
        import numpy as np 
        all_ests = [self.base_estimator] + list(self.residual_estimators)
        return np.array([est.predict(X) for est in all_ests]).T



class DictFlattener(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        import numpy as np
        l = []
        for x in X:
            dic = dict()
            for a in x: 
                #x: dict like {key1:value1, key2: {kkey3: value3, kkey4: value4}}
                #a: entry like key1:value1 or key2: {kkey3: value3, kkey4: value4}
                #dic[a] = 1
                value1 = x[a]
                #print type(value1)

                if isinstance(value1, bool):
                        if value1 == True:
                            dic[a] = int(1)
                        else:
                            dic[a] = int(0)
                        continue

                if isinstance(value1, dict): 
                    for b in value1:
                        value2 = value1[b]
                        #print type(value2)
                        if isinstance(value2, bool):
                            if value2 == True:
                                dic[a+'_'+b] = int(1)
                            else:
                                dic[a+'_'+b] = int(0)
                            continue

                        if isinstance(value2, dict): 
                            print 'Error: more than 2 layers of dict !'
                        else:
                            if isinstance(value2, str): #type(value2)==type(''):
                                dic[a+'_'+b+'_'+value2] = 1
                            if isinstance(value2, (int, float)):
                                dic[a+'_'+b] = value2

                else:
                    if isinstance(value1, str): #(value1)==type(''):
                        dic[a+'_'+value1] = 1
                    if isinstance(value1, (int, float)):
                        dic[a] = value1
            l.append(dic)
        return l


#=====================================================================


ensemble_pipeline = Pipeline([ 
        ('dictflat', DictFlattener() ),
        ('vector', DictVectorizer(sparse=False)),
        ('ensemble', EnsembleTransformer(
                linear_model.LinearRegression(),
                (neighbors.KNeighborsRegressor(n_neighbors=10),
                 ensemble.RandomForestRegressor(min_samples_leaf=5)))),
        ('blend', linear_model.LinearRegression())
       ])



myfile='CAR_PRICE_DATA_1.csv'
df = pd.read_csv(myfile) #, low_memory=False

make = 'toyota'
model = 'camry'

df1 = df[df.MODEL == model]

X_dict = df_Dict_Transform(df1)
y = df1['PRICE'].values

ensemble_pipeline.fit(X_dict, y)

y_pred = ensemble_pipeline.predict(X_dict)


import dill
dill.dump(ensemble_pipeline, open('ensemble_pipeline_%s_%s.dill' %(make, model), 'w'))


from mpl_toolkits.mplot3d import Axes3D

price = df1.ix[:,'PRICE'].values
mile = df1.ix[:,'MILES'].values
year = df1.ix[:,'YEAR'].values
yr = year.tolist()
mi = map(int, mile.tolist())
pr = price.tolist()

fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(yr, mi, pr, c = 'b', marker='o',alpha=0.3)
ax.scatter(yr, mi, y_pred, c = 'r', marker='+',alpha=0.5)

ax.set_title('Toyota Camry')
ax.set_xlabel('Model Year')
ax.set_ylabel('Miles')
ax.set_zlabel('Price ($)')
ax.set_xlim3d(1995, 2018)
ax.set_ylim3d(0, 400000)
ax.set_zlim3d(0, 28000)
plt.show()











