import numpy as np
import pandas as pd
from tqdm import tqdm
import csv, logging, re
from sklearn.svm import SVC
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
import addon

def feature_selection(y,k_val):
	data = pd.read_csv('traindata.txt',sep=' ')
	#data.info()
	best_indices = SelectKBest(f_classif, k=k_val).fit(data, y).get_support(indices=True)
	
	return best_indices
	
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
	for x in range(50):
		best_features = set()
		for cls in train.classes:
			
			features = feature_selection(train._get_binary(cls), x+1)
			best_features.update(features)

		best =  list(best_features)
		X_train = train.X[:,best]
		X_test = test.X[:,best]

		model = SVC(kernel="linear", probability=True)
		model.fit(X_train, y_train)
		results = model.predict(X_test)
		res = zip(results, y_test)
		
		a =  accuracy_score(y_test, results)
		accuracy.append(a)
		print classification_report(y_test, results)

	print "max Accuracy for SelectKbest :", (np.max(accuracy)*100)
	print "Iteration no of Max accuracy:", np.argmax(accuracy)
	



if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	train = addon.Data('data/Training_res.txt', 'data/Training_cls.txt', 'train')
	test = addon.Data('data/Test_res.txt', 'data/Test_cls.txt', 'test')

	run_test(train, test)

