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



def convertdf(data,feature_names):
	normalizer = preprocessing.StandardScaler().fit(data)
	
	data= normalizer.transform(data)
		
	df = pd.DataFrame(data)
	df = df.transpose()
	df.info()
	headers=list();
	for i in range(1081):
		headers.append(i)
	df.columns = headers
	df.to_csv(r'traindata.txt',sep=' ', index=False, header=True)
	

class Data(object):
	"""Class responsible for interfacing with our data, eg) getting the data, stats, etc.."""

	def __init__(self, res_path, cls_path, dataType):
		self.dataType = dataType
		self._get_classes(cls_path)
		self._get_tumor_samples(res_path)
		self._clean()

	def _get_classes(self, path):
		print("Getting " + self.dataType + " classes")
		with open(path, 'r') as f:
			reader = [l.strip() for l in tqdm(f.readlines())]
			self.number_of_samples = reader[0].split(' ')[0]
			self.number_of_classes = reader[0].split(' ')[1]
			self.classes = reader[1].split(' ')[0:]
			self.Y = reader[2].split(' ')
			

	def _get_tumor_samples(self, path):
		print("Getting " + self.dataType + " samples")
		with open(path, 'r') as inputFile:
			lines = [l.strip().split('	') for l in tqdm(inputFile.readlines())]
			data = np.matrix(lines[3:])
			self.feature_names = data[:,1]
			data = data[:,2:]
			data = np.delete(data, list(range(1, data.shape[1], 2)), axis=1)
			if self.dataType=='train':
				convertdf(data,self.feature_names)
		self.X = data.astype(float).T 
		

	def _get_binary(self, name):
		try:
			index = self.classes.index(name) - 1
			return  [c == str(index) for c in self.Y]
		except ValueError:
			return False

	def _describe(self):
		print ("\n------ data " + self.dataType + " description -----")
		print ("X len = ", len(self.X))
		print ("Y len = ", len(self.Y))
		print ("# samples = ", self.number_of_samples)
		print ("# classes = ", self.number_of_classes)
		print ("-----------------------------\n")

	def _clean(self):
		invalid = np.where(np.isin(self.Y, ['14']))[0]
		print (invalid)
		self.Y = np.delete(self.Y, invalid, 0)
		self.X = np.delete(self.X, invalid, 0)
