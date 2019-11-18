import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from itertools import chain
from nltk.corpus import words
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import sys
import numpy as np

with open('./train.txt') as trainfile:
    lines = list(line.rstrip() for line in trainfile)

with open('./test.txt') as testfile:
	testlines =  list(line.rstrip() for line in testfile)

#train
SentenceList=[]
lastspace = 0
for i in range(len(lines)):
	if not lines[i]:
		sentence=""
		for j in range(lastspace,i,1):
			sentence+=lines[j]+","
		SentenceList.append(sentence)
		lastspace = i+1

#test
SentenceList_test=[]
lastspace = 0
for i in range(len(testlines)):
	if not testlines[i]:
		sentence=""
		for j in range(lastspace,i,1):
			sentence+=testlines[j]+","
		SentenceList_test.append(sentence)
		lastspace = i+1

#train
SentencenestedList = []
for setence in SentenceList:
	sent = setence.split(',')
	sent.pop(len(sent)-1)
	SentencenestedList.append(sent)

#test
SentencenestedList_test = []
for setence in SentenceList_test:
	sent = setence.split(',')
	sent.pop(len(sent)-1)
	SentencenestedList_test.append(sent)


#train
allsentences = []
processedList = []
for sentencelist in SentencenestedList:
	sentencestring=''
	nertagstring=''
	postagstring=''
	sentstring=''
	singlesentenceList = []
	for item in sentencelist:
		if item.strip():
			itemlist = item.split(" ")
			sentstring+=itemlist[0]+','
			sentencestring= itemlist[0]+','
			if len(itemlist)<3 or not itemlist[1].isalpha():
				postagstring='O'+','
			else:
				postagstring= itemlist[1]+','

			if len(itemlist)<3:
				nertagstring='O'
			else:
				nertagstring= itemlist[len(itemlist)-1]
			processedList.append(str(sentencestring)+str(nertagstring))
	allsentences.append(sentstring)

#test
allsentences_test = []
processedList_test = []
for sentencelist in SentencenestedList_test:
	sentencestring=''
	nertagstring=''
	postagstring=''
	sentstring=''
	singlesentenceList = []
	for item in sentencelist:
		if item.strip():
			itemlist = item.split(" ")
			sentstring+=itemlist[0]+','
			sentencestring= itemlist[0]+','
			if len(itemlist)<3 or not itemlist[1].isalpha():
				postagstring='O'+','
			else:
				postagstring= itemlist[1]+','

			if len(itemlist)<3:
				nertagstring='O'
			else:
				nertagstring= itemlist[len(itemlist)-1]
			processedList_test.append(str(sentencestring)+str(nertagstring))
	allsentences_test.append(sentstring)



#train
for i in range(len(processedList)):
	processedList[i] = processedList[i].split(',')

#add the word to the vocablist and append none for it:

vocablistdict = {}
for item in processedList:
	if item[0] not in vocablistdict:
		templist = []
		templist.append(item[1])
		vocablistdict[item[0]]=templist


duplicate_train_vocablist = vocablistdict.copy()
for word in vocablistdict:
	for i in range(0,1,1):
		if wn.synsets(word):
			if len(wn.synsets(word)[i].hypernyms())>0:
				# print(wn.synsets(word)[i].hypernyms())
				hypernym = wn.synsets(word)[i].hypernyms()[0].name().split('.')[0]
				if hypernym not in duplicate_train_vocablist:
					duplicate_train_vocablist[hypernym] = ['hypernym']
			else:
				hypernym = None
		if wn.synsets(word):
			if len(wn.synsets(word)[i].lemmas())>0:
				lemma = wn.synsets(word)[i].lemmas()[0].name()
				if lemma not in duplicate_train_vocablist:
					duplicate_train_vocablist[lemma] = ['lemma']
			else:
				lemma = None

		if wn.synsets(word):
			if len(wn.synsets(word)[i].hyponyms())>0:
				hyponym = wn.synsets(word)[i].hyponyms()[0].name().split('.')[0]
				if hyponym not in duplicate_train_vocablist:
					duplicate_train_vocablist[hyponym] = ['hyponym']
			else:
				hyponym = None

		if wn.synsets(word):
			if len(wn.synsets(word)[i].part_meronyms())>0:
				meronym = wn.synsets(word)[i].part_meronyms()[0].name()
				if meronym not in duplicate_train_vocablist:
					duplicate_train_vocablist[meronym] = ['meronym']
			else:
				meronym = None

		if wn.synsets(word):
			if len(wn.synsets(word)[i].part_holonyms())>0:
				holonym = wn.synsets(word)[i].part_holonyms()[0].name()
				if holonym not in duplicate_train_vocablist:
					duplicate_train_vocablist[holonym] = ['holonym']
			else:
				holonym = None


#test
for i in range(len(processedList_test)):
	processedList_test[i] = processedList_test[i].split(',')

vocablistdict_test = {}
for item in processedList_test:
	if item[0] not in vocablistdict_test:
		templist = []
		templist.append(item[1])
		vocablistdict_test[item[0]]=templist

duplicate_test_vocablist = vocablistdict_test.copy()
for word in vocablistdict_test:
	for i in range(0,1,1):
		if wn.synsets(word):
			if len(wn.synsets(word)[i].hypernyms())>0:
				# print(wn.synsets(word)[i].hypernyms())
				hypernym = wn.synsets(word)[i].hypernyms()[0].name().split('.')[0]
				if hypernym not in duplicate_train_vocablist:
					duplicate_test_vocablist[hypernym] = ['hypernym']
			else:
				hypernym = None
		if wn.synsets(word):
			if len(wn.synsets(word)[i].lemmas())>0:
				lemma = wn.synsets(word)[i].lemmas()[0].name()
				if lemma not in duplicate_train_vocablist:
					duplicate_test_vocablist[lemma] = ['lemma']
			else:
				lemma = None

		if wn.synsets(word):
			if len(wn.synsets(word)[i].hyponyms())>0:
				hyponym = wn.synsets(word)[i].hyponyms()[0].name().split('.')[0]
				if hyponym not in duplicate_train_vocablist:
					duplicate_test_vocablist[hyponym] = ['hyponym']
			else:
				hyponym = None

		if wn.synsets(word):
			if len(wn.synsets(word)[i].part_meronyms())>0:
				meronym = wn.synsets(word)[i].part_meronyms()[0].name()
				if meronym not in duplicate_train_vocablist:
					duplicate_test_vocablist[meronym] = ['meronym']
			else:
				meronym = None

		if wn.synsets(word):
			if len(wn.synsets(word)[i].part_holonyms())>0:
				holonym = wn.synsets(word)[i].part_holonyms()[0].name()
				if holonym not in duplicate_train_vocablist:
					duplicate_test_vocablist[holonym] = ['holonym']
			else:
				holonym = None

		

def generatepostags(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


wn_lemmas = set(wn.all_lemma_names())

# extract postags train
for sentence in allsentences:
	postags = generatepostags(sentence)
	for item in postags:
		if item[0] in vocablistdict and item[1].isalpha():
			vocablistdict.get(item[0]).append(item[1])
		
for word in vocablistdict:
	if len(vocablistdict.get(word))==1:
		vocablistdict.get(word).append('None')

#extract postags test
for sentence in allsentences_test:
	postags = generatepostags(sentence)
	for item in postags:
		if item[0] in vocablistdict_test and item[1].isalpha():
			vocablistdict_test.get(item[0]).append(item[1])
		
for word in vocablistdict_test:
	if len(vocablistdict_test.get(word))==1:
		vocablistdict_test.get(word).append('None')



#train set preparartion
unique_tags = []
unique_postags = []

for word in vocablistdict:
	if vocablistdict.get(word)[0] not in unique_tags:
		unique_tags.append(vocablistdict.get(word)[0])
	if vocablistdict.get(word)[1] not in unique_postags:
		unique_postags.append(vocablistdict.get(word)[1])



wordmappingdict_train = {}
counter = 0
for word in duplicate_train_vocablist:
	wordmappingdict_train[word]=counter
	counter=counter+1

posmappingdict = {}
counter = 0
for tag in unique_postags:
	posmappingdict[tag] = counter
	counter = counter+1


tagmappingdict = {}
counter = 0
for nertag in unique_tags:
	tagmappingdict[nertag] = counter
	counter = counter+1


X_train = []
Y_train = []
for word in vocablistdict:
	word_vector=np.zeros(len(duplicate_train_vocablist)+len(posmappingdict))
	postag = vocablistdict.get(word)[1]
	word_vector[wordmappingdict_train.get(word)] = 1
	word_vector[len(duplicate_train_vocablist)+posmappingdict.get(postag)] = 1
	for i in range(0,1,1):
		if wn.synsets(word):
			if len(wn.synsets(word)[i].hypernyms())>0:
				hypernym = wn.synsets(word)[i].hypernyms()[0].name().split('.')[0]
				if hypernym in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(hypernym)] = 1
		if wn.synsets(word):
			if len(wn.synsets(word)[i].lemmas())>0:
				lemma = wn.synsets(word)[i].lemmas()[0].name()
				if lemma in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(lemma)] = 1
			

		if wn.synsets(word):
			if len(wn.synsets(word)[i].hyponyms())>0:
				hyponym = wn.synsets(word)[i].hyponyms()[0].name().split('.')[0]
				if hyponym in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(hyponym)] = 1
			

		if wn.synsets(word):
			if len(wn.synsets(word)[i].part_meronyms())>0:
				meronym = wn.synsets(word)[i].part_meronyms()[0].name()
				if meronym in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(meronym)] = 1

		if wn.synsets(word):
			if len(wn.synsets(word)[i].part_holonyms())>0:
				holonym = wn.synsets(word)[i].part_holonyms()[0].name()
				if holonym in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(holonym)] = 1
	X_train.append(word_vector)
	nertag = vocablistdict.get(word)[0]
	Y_train.append(tagmappingdict.get(nertag))


#test set preparartion
unique_tags_test = []
unique_postags_test = []

for word in vocablistdict_test:
	if vocablistdict_test.get(word)[0] not in unique_tags_test:
		unique_tags_test.append(vocablistdict_test.get(word)[0])
	if vocablistdict_test.get(word)[1] not in unique_postags_test:
		unique_postags_test.append(vocablistdict_test.get(word)[1])



wordmappingdict = {}
counter = 0
for word in duplicate_test_vocablist:
	wordmappingdict[word]=counter
	counter=counter+1

posmappingdict = {}
counter = 0
for tag in unique_postags_test:
	posmappingdict[tag] = counter
	counter = counter+1


tagmappingdict = {}
counter = 0
for nertag in unique_tags_test:
	tagmappingdict[nertag] = counter
	counter = counter+1


X_test = []
Y_test = []
for word in vocablistdict_test:
	word_vector=np.zeros(len(duplicate_train_vocablist)+len(posmappingdict))
	postag = vocablistdict_test.get(word)[1]
	if word in wordmappingdict_train:
		word_vector[wordmappingdict_train.get(word)] = 1
	word_vector[len(duplicate_train_vocablist)+posmappingdict.get(postag)] = 1
	for i in range(0,1,1):
		if wn.synsets(word):
			if len(wn.synsets(word)[i].hypernyms())>0:
				hypernym = wn.synsets(word)[i].hypernyms()[0].name().split('.')[0]
				if hypernym in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(hypernym)] = 1
		
		if wn.synsets(word):
			if len(wn.synsets(word)[i].lemmas())>0:
				lemma = wn.synsets(word)[i].lemmas()[0].name()
				if lemma in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(lemma)] = 1
			

		if wn.synsets(word):
			if len(wn.synsets(word)[i].hyponyms())>0:
				hyponym = wn.synsets(word)[i].hyponyms()[0].name().split('.')[0]
				if hyponym in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(hyponym)] = 1
			

		if wn.synsets(word):
			if len(wn.synsets(word)[i].part_meronyms())>0:
				meronym = wn.synsets(word)[i].part_meronyms()[0].name()
				if meronym in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(meronym)] = 1

		if wn.synsets(word):
			if len(wn.synsets(word)[i].part_holonyms())>0:
				holonym = wn.synsets(word)[i].part_holonyms()[0].name()
				if holonym in wordmappingdict_train:
					word_vector[wordmappingdict_train.get(holonym)] = 1
	X_test.append(word_vector)
	nertag = vocablistdict_test.get(word)[0]
	Y_test.append(tagmappingdict.get(nertag))



clf = LogisticRegression(random_state=0,solver='lbfgs',max_iter=2000,multi_class='multinomial').fit(X_train, Y_train)
y_pred = clf.predict(X_test)

precision, recall, fscore, support = score(Y_test, y_pred,average=None,labels=list(tagmappingdict.values()))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))





#
