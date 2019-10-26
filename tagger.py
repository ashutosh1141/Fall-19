import os
import io
import sys

class Tagger:
	stateList=[]
	taggerDict={}
	tagStartDict={}
	tagEndDict={}

	def __init__(self):
		self.initial_tag_probability = None
		self.transition_probability = None
		self.emission_probability = None

	def addOneSmoothing(self,sentences):
		vocablist=[]
		for sentence in sentences:
			for i in range(0,len(sentence),1):
				if sentence[i][0] not in vocablist:
					vocablist.append(sentence[i][0])
		return len(vocablist)


	def load_corpus(self, path):
		if not os.path.isdir(path):
			sys.exit("Input path is not a directory")
		final_list=[]
		for filename in os.listdir(path):
			if not filename.startswith('.'):
				filename = os.path.join(path, filename)
				try:
					with open(filename) as f:
						for line in f:
							linelist=[]
							if not len(line.strip())==0:
								filecontent=line
								templist=filecontent.split(' ')
								for i in range(0,len(templist),1):
									if templist!='\n':
										items=templist[i].split('/')
										linelist.append(tuple(items))
							if len(linelist)!=0:
								if linelist[len(linelist)-1][0]=='\n':
									linelist = linelist[:-1]
								final_list.append(linelist)
			
					
				except IOError:
					sys.exit("Cannot read file")
		return final_list

	def initialize_probabilities(self, sentences):
		# vocablength=self.addOneSmoothing(sentences)
		print('initial_tag_probability')
		if type(sentences) != list:
			sys.exit("Incorrect input to method")
		#initial_tag_probability
		tagList=[]
		tagCountDict={}
		
		
		for sentence in sentences:
			for i in range(0,len(sentence),1):
				if sentence[i][1] not in tagList:
					tagList.append(sentence[i][1])
		
		dicttag = {}
		for tag in tagList:
			for sentence in sentences:
				if sentence[0][1]==tag:
					if tag not in dicttag:
						dicttag[tag]=1
					else:
						dicttag[tag]+=1

		# print(dicttag)
		self.tagStartDict=dicttag

		tagender={}
		for tag in tagList:
			for sentence in sentences:
				if sentence[len(sentence)-1][1]==tag:
					if tag not in tagender:
						tagender[tag]=1
					else:
						tagender[tag]+=1
		self.tagEndDict=tagender
		# print(self.tagEndDict)


		self.stateList=tagList
		initial_tag_probability_list=[]
		for tag in tagList:
			tagCountAtBeginning=0
			tagCountTotal=0
			for sentence in sentences:
				for i in range(0,len(sentence),1):
					if i==0:
						if sentence[i][1]==tag:
							tagCountAtBeginning=tagCountAtBeginning+1
					if sentence[i][1]==tag:
						tagCountTotal=tagCountTotal+1
			tagCountDict[str(tag)]=tagCountTotal
			probabilityvalueinit=float(tagCountAtBeginning+1)/float(tagCountTotal+len(self.stateList))
			
			templist=[tag,probabilityvalueinit]
			initial_tag_probability_list.append(tuple(templist))
		self.taggerDict=tagCountDict
		self.initial_tag_probability=initial_tag_probability_list
		# print(self.initial_tag_probability)

		#transition probability
		print('transition_probability_list')
		transition_probability_list={}
		for tag in tagList:
			for i in range(len(tagList)):
				fromtagCount=0
				combinationTagCount=0
				for sentence in sentences:
					for j in range(0,len(sentence)-1,1):
						if sentence[j][1]==tag and sentence[j+1][1]==tagList[i]:
							combinationTagCount=combinationTagCount+1
				for sentence in sentences:
					for k in range(0,len(sentence),1):
						if sentence[k][1]==tag:
							fromtagCount=fromtagCount+1
				transition_string=str(tag)+'->'+str(tagList[i])
				probabilityvaluetrans = float(combinationTagCount+1)/float(fromtagCount+len(self.stateList))
				
				transition_probability_list[transition_string]=probabilityvaluetrans
		self.transition_probability=transition_probability_list
		


		# emission_probability
		print('vocablist')
		vocablist=set([])
		for sentence in sentences:
			for i in range(0,len(sentence),1):
					vocablist.add(sentence[i][0])
		# print(len(vocablist))
		dictmap={}
		for sentence in sentences:
			for i in range(0,len(sentence),1):
				if str(sentence[i]) not in dictmap:
					dictmap[str(sentence[i])]=1
				else:
					dictmap[str(sentence[i])]=dictmap[str(sentence[i])]+1
		

		print('emission_probability_list')
		emission_probability_list={}
		for word in vocablist:
			for tag in tagList:
				if str(tuple([word,tag])) in dictmap:
					correctmappings=dictmap[str(tuple([word,tag]))]+1
					emprob=float(correctmappings)/float(tagCountDict[str(tag)]+len(self.stateList))
				else:
					correctmappings=1
					emprob=float(correctmappings)/float(tagCountDict[str(tag)]+len(self.stateList))
				
				emstring = str(word) +'->' + str(tag)
				emission_probability_list[emstring]=emprob

		self.emission_probability=emission_probability_list
		

	def viterbi_decode(self,sentence):
		observation = sentence.split(' ')
		viterbi = []
		backpointer = []
		for i in range(0,len(self.stateList),1):
			templist=[]
			templist1=[]
			for j in range(0,len(observation)+1,1):
				templist.append(0)
				templist1.append('0')
			viterbi.append(templist)
			backpointer.append(templist1)

		for l in range(0,len(self.stateList),1):
			wordstr = str(observation[0]) + '->' +str(self.stateList[l])
			emission_prob = float(self.emission_probability[wordstr])
			trasition_prob = float(float(self.tagStartDict[self.stateList[l]])/float(self.taggerDict[self.stateList[l]]))
			viterbi[l][0] = emission_prob*trasition_prob
			backpointer[l][0] = 'start'
		# print(self.stateList)

		final_sequence_string=[]

		for i in range(1,len(observation),1):
			for j in range(0,len(self.stateList),1):
				wordstr = str(observation[i]) + '->' +str(self.stateList[j])
				maxprobability=0
				# backpointer_value=0
				backpointer_setter=0
				# sequence_string_to_append=[]
				for k in range(0,len(self.stateList),1):
					if wordstr in self.emission_probability:
						value=float(viterbi[k][i-1]*(self.transition_probability[str(self.stateList[k])+'->'+str(self.stateList[j])])*self.emission_probability[wordstr])
					else:
						multiplier=float(1)/float(self.taggerDict[str(self.stateList[j])]+len(self.stateList))
						value = float(viterbi[k][i-1]*(self.transition_probability[str(self.stateList[k])+'->'+str(self.stateList[j])])*multiplier)
					if value>maxprobability:
						maxprobability=value
						backpointer_setter=str(self.stateList[k])
				viterbi[j][i]=maxprobability
				backpointer[j][i]=backpointer_setter

				
				
		maxprobability=0
		finalState=None
		for i in range(0,len(self.stateList),1):
			trasition_prob = float(float(self.tagEndDict[self.stateList[i]])/float(self.taggerDict[self.stateList[i]]))
			value = viterbi[i][len(observation)-1]*trasition_prob
			if value>maxprobability:
				maxprobability=value
				# print(self.stateList[i])
				viterbi[i][len(observation)]=maxprobability
				backpointer[i][len(observation)]=str(self.stateList[i])
				finalState = backpointer[i][len(observation)]
			# print(self.stateList[i])

		

		sequence=[]
		sequence.append(finalState)

		counter=len(observation)
		currentTag = finalState
		for i in range(len(backpointer[0])-2,0,-1):
			index = self.stateList.index(currentTag)
			# print(index)
			# print(i)
			if backpointer[index][i]!='start':
				sequence.append(backpointer[index][i])
				currentTag = backpointer[index][i]
			else:
				break
		sequence.reverse()
		print(sequence)

		return sequence




taggerObject=Tagger()
corplist=taggerObject.load_corpus('./brown_modified')
taggerObject.initialize_probabilities(corplist)
taggerObject.viterbi_decode("The Secretariat is expected to race tomorrow .")
taggerObject.viterbi_decode("People continue to enquire the reason for the race for outer space .")










