from bs4 import BeautifulSoup
import difflib
from nltk import bigrams
import re
import pandas as pd
import pickle

# Use Pandas to read in the training and test data
#train = pd.read_csv("../input/train.csv").fillna("")
#test  = pd.read_csv("../input/test.csv").fillna("")

def build_query_correction_map(print_different=True):
	train = pd.read_csv('./data/train.csv').fillna("")
	test  = pd.read_csv("./data/test.csv").fillna("")
	# get all query
	queries = set(train['query'].values)
	correct_map = {}
	if print_different:
	    print("%30s \t %30s"%('original query','corrected query'))
	for q in queries:
		corrected_q = autocorrect_query(q,train=train,test=test,warning_on=False)
		if print_different and q != corrected_q:
		    print ("%30s \t %30s"%(q,corrected_q))
		correct_map[q] = corrected_q
	return correct_map

def autocorrect_query(query,train=None,test=None,cutoff=0.8,warning_on=True):
	"""
	autocorrect a query based on the training set
	"""	
	if train is None:
		train = pd.read_csv('./data/train.csv').fillna('')
	if test is None:
		test = pd.read_csv('./data/test.csv').fillna('')
	train_data = train.values[train['query'].values==query,:]
	test_data = test.values[test['query'].values==query,:]
	s = ""
	for r in train_data:
		#print "----->r2:",r[2]
		#print "r3:",r[3]
		
		s = "%s %s %s"%(s,BeautifulSoup(r[2]).get_text(" ",strip=True),BeautifulSoup(r[3]).get_text(" ",strip=True))
		#print "s:",s
		#raw_input()
	for r in test_data:
		s = "%s %s %s"%(s,BeautifulSoup(r[2]).get_text(" ",strip=True),BeautifulSoup(r[3]).get_text(" ",strip=True))
	s = re.findall(r'[\'\"\w]+',s.lower())
	print s
	s_bigram = [' '.join(i) for i in bigrams(s)]
	print s_bigram
	raw_input()
	s.extend(s_bigram)
	corrected_query = []	
	for q in query.lower().split():
		print "q:",q
		if len(q)<=2:
			corrected_query.append(q)
			continue
		corrected_word = difflib.get_close_matches(q, s,n=1,cutoff=cutoff)
		print "correction:",corrected_word
		if len(corrected_word) >0:
			corrected_query.append(corrected_word[0])
		else :
			if warning_on:
				print ("WARNING: cannot find matched word for '%s' -> used the original word"%(q))
			corrected_query.append(q)
		print "corrected_query:",corrected_query
		raw_input()
	return ' '.join(corrected_query)


query_map = build_query_correction_map()
with open("query_map.pkl", "w") as f: pickle.dump(query_map, f)
