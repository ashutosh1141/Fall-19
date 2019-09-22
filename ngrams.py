from __future__ import division
import sys
import os
import re

fname='brown_corpus_reviews.txt'

textlist=[]
splittext=[]
sentencelist=[]

S1="Milstein is a gifted violinist who creates all sorts of sounds and arrangements"
S2="It was a strange and emotional thing to be at the opera on a Friday night ."


sentencestovalidate=[]


listS1=S1.split()
listS2=S2.split()

sentencestovalidate.append(listS1)
sentencestovalidate.append(listS2)

vocab=[]
final_count_matrix=[]
final_count_matrix_smoothing=[]
final_matrix=[]
final_matrix_smoothing=[]

def predictSentenceProbability(ngram,smoothing):
	if os.path.isfile(fname):
		with open(fname) as fp:
			textlist=fp.readlines()
	
	if len(textlist)>0:
		for item in textlist:
			splittext.append(item.split())
	
	for item in splittext:
		temparray=[]
		
		for strings in item:
			temparray.append(strings.split())
		sentarray=[]
		
		for sublist in temparray:
			for item in sublist:
				sentarray.append(item)
		
		sentencelist.append(sentarray)
	
	# print(sentencelist)
	generateNgramsCount(ngram,smoothing)
	



def generateProbabilityMatrix(ngram,smoothing):
	print("probabilitymatrix")
	probabilitymatrix=[]
	if smoothing==0:
		for i in range(0,len(final_matrix),1):
			templlist=[]
			for j in range(0,len(final_matrix[i]),1):
				g = (final_matrix[i][j]/final_count_matrix[i])
				templlist.append(g)
			probabilitymatrix.append(templlist)
			if i==0:
				print("["+str(templlist))
			elif i==len(final_matrix)-1:
				print(str(templlist)+"]")
			else:
				print(templlist)
		#calculate sentence probaility

		probaility=1
		for i in range(0,len(probabilitymatrix)-1,1):
			probaility=probaility*probabilitymatrix[i][i+ngram-1]
		print("sentence probability: ",probaility)


	else:
		for i in range(0,len(final_matrix_smoothing),1):
			templlist=[]
			for j in range(0,len(final_matrix_smoothing[i]),1):
				g = float("{0:.5f}".format(final_matrix_smoothing[i][j]/final_count_matrix_smoothing[i]))
				templlist.append(g)
			probabilitymatrix.append(templlist)
			if i==0:
				print("["+str(templlist))
			elif i==len(final_matrix_smoothing)-1:
				print(str(templlist)+"]")
			else:
				print(templlist)
		probaility=1
		for i in range(0,len(probabilitymatrix)-1,1):
			probaility=probaility*probabilitymatrix[i][i+ngram-1]
		print("sentence probability: ",probaility)

def generateNgramsCount(ngram,smoothing):
	if smoothing==0:
		counter=0
		for item in sentencestovalidate:
			counter=counter+1
			print("N-gram count matrix without smoothing for sentence "+str(counter))
			matrix=generateNgramsmatrix(item,ngram)
			vocab_count=generateVocabCount(0,ngram)
			# print(vocab_count)
			count_matrix=generateVocabCount(1,ngram,item)
			print(matrix)
			# print(count_matrix) 
			global final_matrix
			final_matrix=matrix
			global final_count_matrix
			final_count_matrix=count_matrix
			generateProbabilityMatrix(ngram,smoothing)
	
	else:
		vocab_count=generateVocabCount(0,ngram)
		counter=0
		for item in sentencestovalidate:
			counter=counter+1
			print("N-gram count matrix with smoothing for sentence "+str(counter))
			matrix=generateNgramsmatrix(item,ngram)
			count_matrix=generateVocabCount(1,ngram,item)
			for i in range(0,len(count_matrix),1):
				count_matrix[i]=count_matrix[i]+vocab_count;
			for j in range(0,len(matrix),1):
				for i in range(0,len(matrix[j]),1):
					matrix[j][i]=matrix[j][i]+1
			print(matrix)
			# print(count_matrix)
			global final_matrix_smoothing
			final_matrix_smoothing=matrix
			global final_count_matrix_smoothing
			final_count_matrix_smoothing=count_matrix
			generateProbabilityMatrix(ngram,smoothing)


			
def generateVocabCount(args,ngram,sent=None):
	flat_new_list = []
	regex = re.compile('[@_!#$%^&*()<>?/\|}{~:,.";[]]')
	
	if args==0:
		flat_new_list = []
		for sublist in sentencelist:
			for item in sublist:
				if(regex.search(item) == None):
					flat_new_list.append(item)
		global vocab
		vocab=flat_new_list
		return len(set(flat_new_list))
	
	else:
		countlist=[]
		checklist=[]
		for i in range(0,len(sent)-(ngram-2),1):
			templlist=[]
			for j in range(i,i+ngram-1,1):
				templlist.append(sent[j])
			checklist.append(templlist)
		for sublist in checklist:
			# print(sublist)
			counter=0
			for k in range(0,len(vocab)-(ngram-2),1):
				templist=[]
				for l in range(k,k+len(sublist),1):
					templist.append(vocab[l])
				str1=""
				joined_string1=str1.join(templist)
				str2=""
				joined_string2=str2.join(sublist)
				if joined_string1.lower()==joined_string2.lower():
					counter=counter+1
			countlist.append(counter)
		return countlist



def generateNgramsmatrix(sentlist,ngram):
	matrix =[]
	for i in range(0,len(sentlist)-(ngram-2),1):
		searchlist=[]
		for j in range(i,i+ngram-1,1):
			searchlist.append(sentlist[j])
		for n in range(0,len(sentlist),1):
			searchlist.append(sentlist[n])
			element1=searchlist[0]
			counter=0;
			for sublist in sentencelist:
				for l in range(0,len(sublist)-(ngram-1),1):
					templlist=[]
					for m in range(l,l+len(searchlist),1):
						templlist.append(sublist[m])
					str1=""
					joined_string1=str1.join(templlist)
					str2=""
					joined_string2=str2.join(searchlist)
					if joined_string1.lower()==joined_string2.lower():
						counter=counter+1;
			matrix.append(counter)
			searchlist.remove(searchlist[len(searchlist)-1])
			if len(searchlist)>0 and searchlist[0]!=element1:
				searchlist.reverse()
	
	final_list=to_matrix(matrix,len(sentlist))
	return final_list
	



def to_matrix(l, n):
	return [l[i:i+n] for i in range(0, len(l), n)]

# def reshapeMatrix(matrix,row,column):
# 	returnlist=[]
# 	counter=0
# 	templlist=[]


# 	for i in range(0,len(matrix),1):
		
# 		if counter<column:
# 			templlist.append(matrix[i])
# 			counter=counter+1
		
# 		else:
# 			print(counter)
# 			returnlist.append(templlist)
# 			templlist=[]
# 			counter=0

# 	return returnlist




if __name__ == '__main__':
	if sys.argv[1]=='2' and sys.argv[2]=='0':
		predictSentenceProbability(2,0)
	
	if sys.argv[1]=='2' and sys.argv[2]=='1':
		predictSentenceProbability(2,1)
	
	if sys.argv[1]=='3' and sys.argv[2]=='0':
		predictSentenceProbability(3,0)
	
	if sys.argv[1]=='3' and sys.argv[2]=='1':
		predictSentenceProbability(3,1)
	
