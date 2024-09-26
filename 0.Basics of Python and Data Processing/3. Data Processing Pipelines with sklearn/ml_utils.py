#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

# class SomeDataProcess(BaseEstimator, TransformerMixin):
    
#     def __init__(self,arg1,arg2.....) :
        
#         create attributes here 
        
#     def fit(self,x,y=None):
        
#         # here learning from the data happens 
    
#     def transform(self,x,y=None):
        
#         # here you make changes in the data and return transformed data
        
#     def get_feature_names(self):
        
#         # returns the names of the columns going out of transform

# Var Selector 1, Custom Func 4, Convert to Numeric 2, Create Dummies 5, Impute Missing 3 

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.6f} (std: {1:.6f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

class VarSelect(BaseEstimator, TransformerMixin):
    
    def __init__(self,feature_names=[]) :
        
        self.feature_names=feature_names 
        
    def fit(self,x,y=None):
        
        return self 
    
    def transform(self,x,y=None):
        
        return x[self.feature_names]
        
    def get_feature_names(self):
        
        return self.feature_names
    
class ConvertToNumeric(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=[]

    def fit(self,x,y=None):

        self.feature_names=list(x.columns)

        return self
    
    def transform(self,x,y=None):

        for col in x.columns:

            x[col]=pd.to_numeric(x[col],errors='coerce')
        
        return x
    
    def get_feature_names(self):

        return self.feature_names

class MissingImputation(BaseEstimator,TransformerMixin):
    
    def __init__(self,impute_dict={}):
        
        self.impute_dict=impute_dict
        self.feature_names=[]
        
    def fit(self,x,y=None):
        
        self.feature_names=list(x.columns)
        for col in x.columns:
            if col in self.impute_dict:
                pass
            else:
                if x[col].dtype=='O': 
                    self.impute_dict[col]='__missing__'
                else :
                    self.impute_dict[col]=x[col].median()
        return self
    
    def transform(self,x,y=None):
        
        return x.fillna(self.impute_dict)
    
    def get_feature_names(self):
        
        return self.feature_names
    
class CustomFunc(BaseEstimator,TransformerMixin):

    def __init__(self,col_func_dict={}):

        self.col_func_dict=col_func_dict 
        self.feature_names=[]

    def fit(self,x,y=None):

        self.feature_names=list(x.columns)
        return self
    
    def transform(self,x,y=None):

        for col in x.columns:
            x[col]=self.col_func_dict[col](x[col]) 
        
        return x
    
    def get_feature_names(self):

        return self.feature_names
    
class CreateDummies(BaseEstimator,TransformerMixin):

    def __init__(self,freq_percent_cutoff=0.01):

        self.freq_percent_cutoff=freq_percent_cutoff
        self.var_cat_dict={}
        self.feature_names=[]

    def fit(self,x,y=None):

        self.freq_cutoff=x.shape[0]*self.freq_percent_cutoff

        for col in x.columns:

            col_data=x[col].copy()
            k=col_data.value_counts()
            cats_to_be_clubbed=list(k.index[k<=self.freq_cutoff])

            if len(cats_to_be_clubbed)>1:

                col_data=col_data.replace(dict.fromkeys(cats_to_be_clubbed,'__other__'))
                k=col_data.value_counts()
            
            selected_cats=list(k.index[:-1])

            self.var_cat_dict[col]=[cats_to_be_clubbed,selected_cats]
            self.feature_names.extend([col+'_'+str(cat) for cat in selected_cats])
        
        return self
    
    def transform(self,x,y=None):

        out_data={}

        for col in self.var_cat_dict.keys():

            cats_to_be_clubbed=self.var_cat_dict[col][0]
            selected_cats=self.var_cat_dict[col][1]
            out_data[col]=x[col].copy()

            if len(cats_to_be_clubbed)>1:

                out_data[col]=out_data[col].replace(dict.fromkeys(cats_to_be_clubbed,'__other__'))
                

            for cat in selected_cats:

                out_data[col+'_'+str(cat)]=(out_data[col]==cat).astype(int)
            
            del out_data[col]

        return pd.DataFrame(out_data)
    
    def get_feature_names(self):

        return self.feature_names

class DateComponents(BaseEstimator,TransformerMixin):

    def __init__(self,keep_year=False):

        self.feature_names=[]
        self.week_freq=7
        self.month_freq=12
        self.month_day_freq=31
        self.keep_year=keep_year

    def fit(self,x,y=None):

        for col in x.columns:

            for kind in ['week','month','month_day']:

                self.feature_names.extend([col + '_'+kind+temp for temp in ['_sin','_cos']])
            
            if self.keep_year:

                self.feature_names.append(col+'_year')

        return self 

    def transform(self,x):

        out_data={}

        for col in x.columns:

            out_data[col]=pd.to_datetime(x[col])
            
            wdays=out_data[col].dt.dayofweek
            month=out_data[col].dt.month
            day=out_data[col].dt.day

            out_data[col+'_week_sin']=np.sin(2*np.pi*wdays/self.week_freq)
            out_data[col+'_week_cos']=np.cos(2*np.pi*wdays/self.week_freq)

            out_data[col+'_month_sin']=np.sin(2*np.pi*month/self.month_freq)
            out_data[col+'_month_cos']=np.cos(2*np.pi*month/self.month_freq)

            out_data[col+'_month_day_sin']=np.sin(2*np.pi*day/self.month_day_freq)
            out_data[col+'_month_day_cos']=np.cos(2*np.pi*day/self.month_day_freq)

            if self.keep_year:
                out_data[col+'_year']=x[col].dt.year

            del out_data[col]

        return pd.DataFrame(out_data)

    def get_feature_names(self):

        return self.feature_names

class DateDiffs(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=[]

    def fit(self,x,y=None):

        cols=list(x.columns)            
        num_cols=len(x.columns)

        for i in range(num_cols-1):

            for j in range(i+1,num_cols):

                name=cols[i]+'_diff_with_'+cols[j]
                self.feature_names.append(name)

        return self


    def transform(self,x):

        cols=list(x.columns)
        num_cols=len(cols)
        out_data={}

        for col in cols:

            out_data[col]=pd.to_datetime(x[col])

        for i in range(num_cols-1):

            for j in range(i+1,num_cols):

                name=cols[i]+'_diff_with_'+cols[j]
                out_data[name]=(out_data[cols[i]]-out_data[cols[j]]).dt.days

        for col in cols:
            del out_data[col]

        return pd.DataFrame(out_data)

    def get_feature_names(self):

        return self.feature_names
    
class TextFeatures(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=[]
        self.tfidfs={}

    def fit(self,x,y=None):

        for col in x.columns:

            self.tfidfs[col]=TfidfVectorizer(analyzer='word',stop_words='english',
                token_pattern=r'(?u)\b[A-Za-z]+\b',min_df=0.01,max_df=0.8,max_features=200)
            
            self.tfidfs[col].fit(x[col])
            self.feature_names.extend([col+'_'+word for word in list(self.tfidfs[col].get_feature_names_out())])

        return self


    def transform(self,x):

        datasets={}

        for col in x.columns:

            datasets[col]=pd.DataFrame(data=self.tfidfs[col].transform(x[col]).toarray(),
                                       columns=[col+'_'+word for word in list(self.tfidfs[col].get_feature_names_out())])

        return pd.concat(datasets.values(),axis=1)

    def get_feature_names(self):

        return self.feature_names
    
class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step=self.steps[-1][-1]

        return last_step.get_feature_names()
    
class DataPipe:

    def __init__(self,cat_to_dummies=None,
                      cat_to_num=None,
                      simple_num=None,
                      custom_func_dict=None,
                      date_diffs=None,
                      date_components=None,
                      text_feat=None,
                      for_catboost=None):
        
        self.cat_to_dummies=cat_to_dummies 
        self.cat_to_num=cat_to_num
        self.simple_num=simple_num
        self.custom_func_dict=custom_func_dict
        self.date_diffs=date_diffs
        self.date_components=date_components
        self.text_feat=text_feat
        self.for_catboost=for_catboost

    def fit(self,x):

        pipelines={}
        i=1

        if self.cat_to_dummies is not None:

            if self.for_catboost is not None:

                pipelines['p'+str(i)]=pdPipeline([
                                                ('var_select',VarSelect(self.cat_to_dummies)),
                                                ('missing_trt',MissingImputation())
                                                ])

            else:

                pipelines['p'+str(i)]=pdPipeline([
                                                ('var_select',VarSelect(self.cat_to_dummies)),
                                                ('missing_trt',MissingImputation()),
                                                ('create_dummies',CreateDummies())
                                                ])
            i=i+1
        if self.cat_to_num is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                            ('var_select',VarSelect(self.cat_to_num)),
                                            ('convert_to_num',ConvertToNumeric()),
                                            ('missing trt',MissingImputation())
                                            ])
            i=i+1
        if self.simple_num is not None:
            
            pipelines['p'+str(i)]=pdPipeline([
                                            ('var_select',VarSelect(self.simple_num)),
                                            ('missing_trt',MissingImputation())
                                            ])
            i=i+1
        if self.custom_func_dict  is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                            ('var_select',VarSelect(list(self.custom_func_dict.keys()))),
                                            ('custom_func',CustomFunc(self.custom_func_dict)),
                                            ('missing_trt',MissingImputation())
                                            ])
            i=i+1
        
        if self.date_diffs is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                            ('var_select',VarSelect(self.date_diffs)),
                                            ('date_diffs',DateDiffs()),
                                            ('missing_trt',MissingImputation())
                                            ])

            i+=1
        
        if self.date_components is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                            ('var_select',VarSelect(self.date_components)),
                                            ('cyclic_feat',DateComponents()),
                                            ('missing_trt',MissingImputation())
                                            ])

            i+=1
        
        if self.text_feat is not None:

            pipelines['p'+str(i)]=pdPipeline([
                                            ('var_select',VarSelect(self.text_feat)),
                                            ('missing_trt',MissingImputation()),
                                            ('text_feat',TextFeatures())
                                            ])

            i+=1
        
        
        self.data_pipe=FeatureUnion(pipelines.items())
        self.all_feature_names=[]

        self.data_pipe.fit(x)

        for pipe_name,pipe_obj in list(pipelines.items()):

            self.all_feature_names.extend(pipe_obj.get_feature_names())
        
        return self
        
    def transform(self,x):

        mydata=pd.DataFrame(
                            data=self.data_pipe.transform(x),
                            columns=self.all_feature_names
                            )
        if self.for_catboost:
            for col in self.all_feature_names:
                if col in self.cat_to_dummies:pass
                else:
                    mydata[col]=pd.to_numeric(mydata[col],errors='coerce')
                    
        return mydata 
    
class Utils:

    @staticmethod
    def report(results, n_top=3, score_metric='r2'):
    
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results[f'rank_test_{score_metric}'] == i)
            for candidate in candidates:
                mean_score = results[f'mean_test_{score_metric}'][candidate]
                std_score = results[f'std_test_{score_metric}'][candidate]
                params = results['params'][candidate]
                print(f"Model with rank: {i}")
                print(f"Mean validation score for {score_metric}: {mean_score:.6f} (std: {std_score:.6f})")
                print(f"Parameters: {params}\n")