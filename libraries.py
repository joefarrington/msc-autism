import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools

from collections import namedtuple

from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from scikitplot.metrics import plot_confusion_matrix

from imblearn.ensemble import BalancedRandomForestClassifier

#Define a named tuple to store the results of training
Results = namedtuple('Results', 'X_train X_test y_train y_test y_pred model mcc_test acc_test conf')

#Define a function for training models on repeated train/test splits
def repeated_training(X, y, n_repeats, n_folds, param_grid, model, scorer, verbose):
    
    #Create a dictionary to store the results from each run
    results_dict = {}
    
    #Create a dataframe to store all of the test set labels and predictions
    cmdf = pd.DataFrame(columns=['label', 'pred'])

    for i in range(0, n_repeats):

        #Create a train/test split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y)

        #Run 10-fold cross validation on the training set to find the best hyperparamter combination
        #The grid search object will fit a model using the best combination of hyperparamters on 
        #the whole training set
        skf = StratifiedKFold(n_splits=n_folds)
        gs = GridSearchCV(model, param_grid, cv=skf, iid=False, scoring=scorer, n_jobs=-1, verbose=verbose)
        gs.fit(X_train, y_train)

        #Get predictions on the test set
        y_pred = gs.predict(X_test)

        #Get the accuracy and MCC on the test set
        mcc_test = matthews_corrcoef(y_test, y_pred)
        acc_test = accuracy_score(y_test, y_pred)
        
        #Calculate the confusion matrix on the test set
        conf = confusion_matrix(y_test,y_pred)
        
        #Store the labels and predictions in the cumulative dataframe
        tempdf = pd.DataFrame({'label': y_test, 'pred': y_pred})
        cmdf = cmdf.append(tempdf)

        #Create a named tuple to store the results from this run
        results_dict[f'Run{i}'] = Results(X_train, X_test, y_train, y_test, y_pred, gs, mcc_test, acc_test, conf)
        
        print(f'Run {i} completed')
        
    #Print out the metrics
    rkeys = results_dict.keys()
    acc = []
    for key in rkeys:
        acc.append(results_dict[key].acc_test)
    print(f'Mean test accuracy: {np.mean(acc)}')
    print(f'Std dev test accuracy: {np.std(acc)}')

    mcc = []
    for key in rkeys:
        mcc.append(results_dict[key].mcc_test)
    print(f'Mean test MCC: {np.mean(mcc)}')
    print(f'Std dev test MCC: {np.std(mcc)}')
    
    results_dict["cmdf"] = cmdf 

    #Return the results dictionary
    return(results_dict)



#Define a function to print the results from a results dictionary
def print_test_res(results_dict, n_repeats):
    
    #Accuracy
    accs = []
    for i in range(0,n_repeats):
        accs.append(results_dict[f'Run{i}'].acc_test)
    print(f'Mean test accuracy: {np.mean(accs)}')
    print(f'Std dev test accuracy: {np.std(accs)}')
    
    #MCC
    mccs = []
    for i in range(0,n_repeats):
        mccs.append(results_dict[f'Run{i}'].mcc_test)
    print(f'Mean test MCC: {np.mean(mccs)}')
    print(f'Std dev test MCC: {np.std(mccs)}')

#Define a function to plot a normalized confusion matrix
def plot_normalized_cm(y_true, y_pred, labels, ax):
    #Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    #Get the totals for each row
    row_tot = np.sum(cm, axis=1).reshape(-1,1)
    #Normalize the matrix so that each row sums to 1
    cm_n = cm/row_tot
    #Plot a seaborn heatmap
    hm = sns.heatmap(cm_n, cmap='Blues', square=True, annot=cm_n, vmin=0, vmax=1, fmt = ".2f", xticklabels = labels , yticklabels= labels, cbar=False, ax=ax)
    #Return the heatmap
    return hm


###This code for pickling large files on Mac OS X was taken from
### https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
### and was submitted by user Sam Cohan in April 2017

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

### End of code written by Sam Cohan


