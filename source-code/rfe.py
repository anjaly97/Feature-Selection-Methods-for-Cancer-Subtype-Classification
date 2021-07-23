import numpy as np
import pandas as pd
from tqdm import tqdm
import csv, logging, re
from sklearn.svm import SVC
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
import addon


def feature_selection(model,y):
	data = pd.read_csv('traindata.txt',sep=' ')
	#data.info()
	rfe_selector = RFE(estimator=model,n_features_to_select = 504, step = 1)
	rfe_selector.fit(data, y)
	return data.columns[rfe_selector.get_support()]

def run_test(train, test):
	train._describe()
	test._describe()

	normalizer = preprocessing.StandardScaler().fit(train.X)
	train.X = normalizer.transform(train.X)
	test.X = normalizer.transform(test.X)
	# ========================
	#    System parameters
	# ========================
	y_train = train.Y
	y_test  = test.Y	
	X_train = train.X
	X_test = test.X
	
	accuracy = list()
	model = SVC(kernel="linear", probability=True)
	best_features = set()
		
	features = feature_selection(model,train.Y)
	best_features.update(features)

	bst =  list(best_features)
	best = list()
	for i in features:
		best.append(int(i))

		  
	X_train = train.X[:,best]
	X_test = test.X[:,best]

	
	model.fit(X_train, y_train)
	results = model.predict(X_test)
	res = zip(results, y_test)
	
	accuracy=list()	
	a =  accuracy_score(y_test, results)
	accuracy.append(a)
	print (classification_report(y_test, results))

	print "max Accuracy for RFE :",100*np.max(accuracy)
	
	



if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	train = addon.Data('data/Training_res.txt', 'data/Training_cls.txt', 'train')
	test = addon.Data('data/Test_res.txt', 'data/Test_cls.txt', 'test')

	run_test(train, test)

