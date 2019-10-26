import tarfile
import sys
import os
import hashlib
import numpy as np
from numpy import array
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F


# Task 1: Load the data
# For this task you will load the data, create a vocabulary and encode the reviews with integers
DATA = []
LABELS = []
TOKENDICT = {}

def read_file_test(path_to_dataset):

    if not tarfile.is_tarfile(path_to_dataset):
        sys.exit("Input path is not a tar file")
    
    dirent = tarfile.open(path_to_dataset)
    data = []
    labels = []
    
    for member in dirent.getmembers():
        f = dirent.extractfile(member)
        member_name = str(member).split(' ')[1]
        member_name_list = member_name.split('/')
        if f is not None and 'test' in member_name_list:
            content = f.read()
            sentimentList = str(member).split(' ')[1].split('/')
            if len(sentimentList)==4 and not sentimentList[3].startswith('.'):
                data.append(content)
                labels.append(sentimentList[2])
    dirent.close()
    return [data,labels]

def read_file(path_to_dataset):

    if not tarfile.is_tarfile(path_to_dataset):
        sys.exit("Input path is not a tar file")
    dirent = tarfile.open(path_to_dataset)
    for member in dirent.getmembers():
        f = dirent.extractfile(member)
        member_name = str(member).split(' ')[1]
        member_name_list = member_name.split('/')
        if f is not None and 'train' in member_name_list:
            content = f.read()
            sentimentList = str(member).split(' ')[1].split('/')
            if len(sentimentList)==4 and not sentimentList[3].startswith('.'):
                global DATA
                DATA.append(content)
                global LABELS
                LABELS.append(sentimentList[2])
    dirent.close()
    return [DATA,LABELS]


def preprocess(text):
    global TOKENDICT
    if type(text) is not list:
        sys.exit("Please provide a list to the method")
    counter = 0
    for item in text:
        reviewList = item.decode('ASCII').split(" ")
        for token in reviewList:
            if token not in TOKENDICT:
                TOKENDICT[token] = counter
                counter = counter + 1
    return TOKENDICT



def encode_review(vocab, text):
    if type(vocab) is not dict or type(text) is not list:
        sys.exit("Please provide a list to the method")
    encodedDataSet = []
    for item in text:
        reviewList = item.decode('ASCII').split(" ")
        encodedlist = []
        for token in reviewList:
            encodedlist.append(vocab[token])

        encodedDataSet.append(encodedlist)
    return encodedDataSet


def encode_labels(labels): # Note this method is optional (if you have not integer-encoded the labels)

    if type(labels) is not list:
        sys.exit("Please provide a list to the method")
    for i in range(len(labels)):
        if labels[i]=='pos':
            labels[i]=1
        else:
            labels[i]=0

    return labels
    


def pad_zeros(encoded_reviews, seq_length = 600):
    if type(encoded_reviews) is not list:
        sys.exit("Please provide a list to the method")
    for i in range(len(encoded_reviews)):
        if len(encoded_reviews[i])>seq_length:
            encoded_reviews[i] = encoded_reviews[i][:seq_length]
        else:
            diff = seq_length-len(encoded_reviews[i])
            for j in range(diff):
                encoded_reviews[i].append(0)

    return encoded_reviews


# Task 2: Load the pre-trained embedding vectors
# For this task you will load the pre-trained embedding vectors from Word2Vec

def load_embedding_file(embedding_file, token_dict):

    if not os.path.isfile(embedding_file):
        sys.exit("Input embedding path is not a file")
    if type(token_dict) is not dict:
        sys.exit("Input a dictionary!")

    embedding_dict = {}
    fileList = []
    embedding_dict_last_value = 0
    with open(embedding_file) as infile:
            for line in infile:
                lineList = line.split(' ')
                if(lineList[0] in token_dict):
                    # print(list(map(float,lineList[1:])))
                    embedding_dict[token_dict[lineList[0]]] = torch.tensor(list(map(float,lineList[1:])))
                    embedding_dict_last_value_shape= torch.tensor(list(map(float,lineList[1:])))
    for token in token_dict:
        if token_dict[token] not in embedding_dict:
            embedding_dict[token_dict[token]] =torch.zeros(embedding_dict_last_value_shape.shape)

    return embedding_dict



# Task 3: Create a TensorDataset and DataLoader

def create_data_loader(encoded_reviews, labels, batch_size = 50):

    if type(encoded_reviews) is not list or type(labels) is not list:
        sys.exit("Please provide a list to the method")

    train_data = TensorDataset(torch.from_numpy(array(encoded_reviews)), torch.from_numpy(array(labels)))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    return train_loader


# Task 4: Define the Baseline model here

# This is the baseline model that contains an embedding layer and an fcn for classification
class BaseSentiment(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim,max_length, n_layers, output_size,args,embedding_matrix,drop_p = 0.5):
        super(BaseSentiment,self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict({'weight': embedding_matrix})
        self.embedding.weight.requires_grad = False
        self.fcn = nn.Linear(max_length, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.contiguous().view(-1, self.hidden_dim)
        out = self.fcn(x)
        out_sig = self.sigmoid(out)
        out_sig = out_sig.view(batch_size, -1)
        sig_out = out_sig[:, -1]
        return sig_out
        

# Task 5: Define the RNN model here
# This model contains an embedding layer, an rnn and an fcn for classification
#LSTM Network
class RNNSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim,max_length, n_layers, output_size,args,embedding_matrix,bidirectional,seq_length,model_args,drop_p = 0.5):
        super(RNNSentiment,self).__init__()
        self.output_size = output_size
        self.model_args = model_args
        self.n_layers = n_layers
        self.hidden_dim = max_length
        self.bidirectional = bidirectional
        self.seq_len = seq_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict({'weight': embedding_matrix})
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, max_length, n_layers, batch_first=True,bidirectional=self.bidirectional)
        self.gru = nn.GRU(embedding_dim,max_length,n_layers,batch_first = True,bidirectional=self.bidirectional)
        self.rnn_cell = nn.RNN(embedding_dim,max_length,n_layers,batch_first = True,bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(drop_p)
        self.fcn = nn.Linear(max_length, output_size)
        self.sigmoid = nn.Sigmoid()

        

    def forward(self,x):
        #lstm model
        if self.model_args==1:
            batch_size = x.size(0)
            x = self.embedding(x)
            if self.bidirectional:
                weights = next(self.parameters()).data
                hidden = (weights.new(2*self.n_layers, batch_size, self.hidden_dim).zero_(),
                          weights.new(2*self.n_layers, batch_size, self.hidden_dim).zero_())
            else:
                weights = next(self.parameters()).data
                hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                          weights.new(self.n_layers, batch_size, self.hidden_dim).zero_())
            x, hidden = self.lstm(x, hidden)
            if self.bidirectional:
                x = x.contiguous().view(-1, self.seq_len, 2, self.hidden_dim)
                lstm_out_bw = x[:, 0, 1, :]
                lstm_out_fw = x[:, -1, 0, :]
                x = torch.add(input=lstm_out_bw, alpha=1, other=lstm_out_fw)
                x = torch.div(x, 2)
            x = self.dropout(x)
            x = x.contiguous().view(-1, self.hidden_dim)
            out = self.fcn(x)
            out_sig = self.sigmoid(out)
            out_sig = out_sig.view(batch_size, -1)
            sig_out = out_sig[:, -1]
            return sig_out, hidden
        
        #gru model
        if self.model_args==2:
            batch_size = x.size(0)
            if self.bidirectional:
                hidden = torch.zeros(2*self.n_layers, batch_size, self.hidden_dim)
            else:
                hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            x = self.embedding(x)
            x, hidden = self.gru(x, hidden)
            if self.bidirectional:
                x = x.contiguous().view(-1, self.seq_len, 2, self.hidden_dim)
                gru_out_bw = x[:, 0, 1, :]
                gru_out_fw = x[:, -1, 0, :]
                x = torch.add(input=gru_out_bw, alpha=1, other=gru_out_fw)
                x = torch.div(x, 2)
            x = self.dropout(x)
            x = x.contiguous().view(-1, self.hidden_dim)
            out = self.fcn(x)
            out_sig = self.sigmoid(out)
            out_sig = out_sig.view(batch_size, -1)
            sig_out = out_sig[:, -1]
            return sig_out, hidden

        #simple vanilla RNN
        if self.model_args==3:
            batch_size = x.size(0)
            if self.bidirectional:
                hidden = torch.zeros(2*self.n_layers, batch_size, self.hidden_dim)
            else:
                hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            x = self.embedding(x)
            x, hidden = self.rnn_cell(x, hidden)
            if self.bidirectional:
                x = x.contiguous().view(-1, self.seq_len, 2, self.hidden_dim)
                vanilla_out_bw = x[:, 0, 1, :]
                vanilla_out_fw = x[:, -1, 0, :]
                x = torch.add(input=vanilla_out_bw, alpha=1, other=vanilla_out_fw)
                x = torch.div(x, 2)
            x = self.dropout(x)
            x = x.contiguous().view(-1, self.hidden_dim)
            out = self.fcn(x)
            out_sig = self.sigmoid(out)
            out_sig = out_sig.view(batch_size, -1)
            sig_out = out_sig[:, -1]
            return sig_out, hidden


# Task 6: Define the RNN model here

# This model contains an embedding layer, self-attention and an fcn for classification
class AttentionSentiment(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(AttentionSentiment,self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.load_state_dict({'weight': weights})
        self.word_embeddings.weight.requires_grad = False
        self.dropout = 0.5
        self.lstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        self.attention1 = nn.Linear(2*hidden_size, 300)
        self.attention2 = nn.Linear(300, 30)
        self.fcn = nn.Linear(30*2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_sentences):
        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        h = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        c = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        output, (h_n, c_n) = self.lstm(input, (h, c))
        output = output.permute(1, 0, 2)
        weight_mat = self.attention2(F.tanh(self.attention1(output)))
        weight_mat = weight_mat.permute(0, 2, 1)
        weight_mat = F.softmax(weight_mat, dim=2)
        hidden_matrix = torch.bmm(weight_mat, output)
        fc_out = self.fcn(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        output = self.sigmoid(fc_out)
        return output

def returnTestAccuracy(net,criterion):
    cumulatedList_test = read_file_test('./movie_reviews.tar.gz')
    DATA_test = cumulatedList_test[0]
    LABELS_test = cumulatedList_test[1]
    TOKENDICT_test = preprocess(DATA_test)
    enodedtSet_test = encode_review(TOKENDICT_test,DATA_test)
    vocabsize_test = len(TOKENDICT_test)
    encodedlabels_test = encode_labels(LABELS_test)
    enodedtSetPaddes_test = pad_zeros(enodedtSet_test)
    testDataSet = create_data_loader(enodedtSetPaddes_test,encodedlabels_test)

    
    num_correct = 0
    #testing for lstm
    for inputs, labels in testDataSet:
        test_output = net.forward(inputs)
        loss = criterion(test_output, labels.float())
        preds = torch.round(test_output.squeeze())
        correct_tensor = preds.eq(labels.float().view_as(preds))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    return num_correct,len(testDataSet.dataset)


#################################################################################
######################## BASELINE MODEL #########################################
#################################################################################

def runBaseLine():
#set hyper_parameters
    cumulatedList = read_file('./movie_reviews.tar.gz')
    DATA = cumulatedList[0]
    LABELS = cumulatedList[1]
    TOKENDICT = preprocess(DATA)
    enodedtSet = encode_review(TOKENDICT,DATA)
    vocabsize = len(TOKENDICT)
    encodedlabels = encode_labels(LABELS)
    enodedtSetPaddes = pad_zeros(enodedtSet)
    max_length = len(enodedtSetPaddes[0])
    embedding_dict = load_embedding_file('./wiki-news-300d-1M.vec',TOKENDICT)
    trainDataSet = create_data_loader(enodedtSetPaddes,encodedlabels)

    embedding_matrix = torch.zeros((vocabsize+1, 300))
    counter = 0
    for word in TOKENDICT:
        embedding_vector = embedding_dict.get(TOKENDICT[word])
        if embedding_vector is not None:
            embedding_matrix[counter] = embedding_vector
            counter = counter+1
    print("training Baseline model")
    vocab_size = vocabsize
    output_size = 1
    learning_rate = 0.1
    n_epochs = 100


    basenet = BaseSentiment(vocabsize+1, 300, 32, 2,1,1,embedding_matrix)
    print(basenet)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(basenet.parameters(), lr=learning_rate)


    #training for Baseline Model
    for epoch in range(1, n_epochs + 1):
            counter = 0
            for inputs, labels in trainDataSet:
                counter += 1
                basenet.zero_grad()
                output= basenet.forward(inputs)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                if counter==38:
                    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                    print("Loss: {:.4f}".format(loss.item()))

    print("testing Baseline model")
    num_correct,length =  returnTestAccuracy(basenet,criterion)
    print("Test Accuracy: {:.2f}".format(num_correct/length))

#################################################################################
######################## BASELINE MODEL #########################################
#################################################################################


#################################################################################
######################## RNN-LSTM MODEL #########################################
#################################################################################

# set hyper_parameters
def runLSTM(bidirectional=False):
    cumulatedList = read_file('./movie_reviews.tar.gz')
    DATA = cumulatedList[0]
    LABELS = cumulatedList[1]
    TOKENDICT = preprocess(DATA)
    enodedtSet = encode_review(TOKENDICT,DATA)
    vocabsize = len(TOKENDICT)
    encodedlabels = encode_labels(LABELS)
    enodedtSetPaddes = pad_zeros(enodedtSet)
    max_length = len(enodedtSetPaddes[0])
    embedding_dict = load_embedding_file('./wiki-news-300d-1M.vec',TOKENDICT)
    trainDataSet = create_data_loader(enodedtSetPaddes,encodedlabels)

    embedding_matrix = torch.zeros((vocabsize+1, 300))
    counter = 0
    for word in TOKENDICT:
        embedding_vector = embedding_dict.get(TOKENDICT[word])
        if embedding_vector is not None:
            embedding_matrix[counter] = embedding_vector
            counter = counter+1

    vocab_size = vocabsize
    output_size = 1
    learning_rate = 0.01
    n_epochs = 2

    net = RNNSentiment(vocabsize+1, 300, 16, 2,1,1,embedding_matrix,bidirectional,600,1)
    print(net)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    #training for lstm
    
    for epoch in range(1, n_epochs + 1):
            counter = 0
            for inputs, labels in trainDataSet:
                counter += 1
                net.zero_grad()
                output, h = net.forward(inputs)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                if counter==38:
                    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                    print("Loss: {:.4f}".format(loss.item()))

    # create test set
    cumulatedList_test = read_file_test('./movie_reviews.tar.gz')
    DATA_test = cumulatedList_test[0]
    LABELS_test = cumulatedList_test[1]
    TOKENDICT_test = preprocess(DATA_test)
    enodedtSet_test = encode_review(TOKENDICT_test,DATA_test)
    vocabsize_test = len(TOKENDICT_test)
    encodedlabels_test = encode_labels(LABELS_test)
    enodedtSetPaddes_test = pad_zeros(enodedtSet_test)
    testDataSet = create_data_loader(enodedtSetPaddes_test,encodedlabels_test)


    
    num_correct = 0

    #testing for lstm
    for inputs, labels in testDataSet:
        test_output, test_h = net.forward(inputs)
        loss = criterion(test_output, labels.float())
        preds = torch.round(test_output.squeeze())
        correct_tensor = preds.eq(labels.float().view_as(preds))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)
        
    print("Test Accuracy: {:.2f}".format(num_correct/len(testDataSet.dataset)))

#################################################################################
######################## RNN-LSTM MODEL #########################################
#################################################################################


#################################################################################
######################## vanilaarnnMODEL ########################################
#################################################################################
def SimpleVanillaRNN(bidirectional=False):
# # set hyper_parameters
    cumulatedList = read_file('./movie_reviews.tar.gz')
    DATA = cumulatedList[0]
    LABELS = cumulatedList[1]
    TOKENDICT = preprocess(DATA)
    enodedtSet = encode_review(TOKENDICT,DATA)
    vocabsize = len(TOKENDICT)
    encodedlabels = encode_labels(LABELS)
    enodedtSetPaddes = pad_zeros(enodedtSet)
    max_length = len(enodedtSetPaddes[0])
    embedding_dict = load_embedding_file('./wiki-news-300d-1M.vec',TOKENDICT)
    trainDataSet = create_data_loader(enodedtSetPaddes,encodedlabels)

    embedding_matrix = torch.zeros((vocabsize+1, 300))
    counter = 0
    for word in TOKENDICT:
        embedding_vector = embedding_dict.get(TOKENDICT[word])
        if embedding_vector is not None:
            embedding_matrix[counter] = embedding_vector
            counter = counter+1
    vocab_size = vocabsize
    output_size = 1
    learning_rate = 0.05
    n_epochs = 20

    net = RNNSentiment(vocabsize+1, 300, 16, 1,1,1,embedding_matrix,bidirectional,600,3)
    print(net)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    #training for lstm
    
    for epoch in range(1, n_epochs + 1):
            counter = 0
            for inputs, labels in trainDataSet:
                counter += 1
                net.zero_grad()
                output, h = net.forward(inputs)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                if counter==38:
                    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                    print("Loss: {:.4f}".format(loss.item()))

    # create test set
    cumulatedList_test = read_file_test('./movie_reviews.tar.gz')
    DATA_test = cumulatedList_test[0]
    LABELS_test = cumulatedList_test[1]
    TOKENDICT_test = preprocess(DATA_test)
    enodedtSet_test = encode_review(TOKENDICT_test,DATA_test)
    vocabsize_test = len(TOKENDICT_test)
    encodedlabels_test = encode_labels(LABELS_test)
    enodedtSetPaddes_test = pad_zeros(enodedtSet_test)
    testDataSet = create_data_loader(enodedtSetPaddes_test,encodedlabels_test)


    
    num_correct = 0
    
    #testing for lstm
    for inputs, labels in testDataSet:
        test_output, test_h = net.forward(inputs)
        loss = criterion(test_output, labels.float())
        preds = torch.round(test_output.squeeze())
        correct_tensor = preds.eq(labels.float().view_as(preds))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)
        
    print("Test Accuracy: {:.2f}".format(num_correct/len(testDataSet.dataset)))

#################################################################################
######################## vanilaarnnMODEL ########################################
#################################################################################


#################################################################################
######################## GRUMODEL ###############################################
#################################################################################
def GRUMODEL(bidirectional=False):
# set hyper_parameters
    cumulatedList = read_file('./movie_reviews.tar.gz')
    DATA = cumulatedList[0]
    LABELS = cumulatedList[1]
    TOKENDICT = preprocess(DATA)
    enodedtSet = encode_review(TOKENDICT,DATA)
    vocabsize = len(TOKENDICT)
    encodedlabels = encode_labels(LABELS)
    enodedtSetPaddes = pad_zeros(enodedtSet)
    max_length = len(enodedtSetPaddes[0])
    embedding_dict = load_embedding_file('./wiki-news-300d-1M.vec',TOKENDICT)
    trainDataSet = create_data_loader(enodedtSetPaddes,encodedlabels)

    embedding_matrix = torch.zeros((vocabsize+1, 300))
    counter = 0
    for word in TOKENDICT:
        embedding_vector = embedding_dict.get(TOKENDICT[word])
        if embedding_vector is not None:
            embedding_matrix[counter] = embedding_vector
            counter = counter+1

    print("training GRU Model")
    vocab_size = vocabsize
    output_size = 1
    learning_rate = 0.01
    n_epochs = 20

    net = RNNSentiment(vocabsize+1, 300, 16, 2,1,1,embedding_matrix,bidirectional,600,2)
    print(net)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    #training for lstm

    for epoch in range(1, n_epochs + 1):
            counter = 0
            for inputs, labels in trainDataSet:
                counter += 1
                net.zero_grad()
                output, h = net.forward(inputs)
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                if counter==38:
                    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                    print("Loss: {:.4f}".format(loss.item()))

    print("testing GRU Model")
    cumulatedList_test = read_file_test('./movie_reviews.tar.gz')
    DATA_test = cumulatedList_test[0]
    LABELS_test = cumulatedList_test[1]
    TOKENDICT_test = preprocess(DATA_test)
    enodedtSet_test = encode_review(TOKENDICT_test,DATA_test)
    vocabsize_test = len(TOKENDICT_test)
    encodedlabels_test = encode_labels(LABELS_test)
    enodedtSetPaddes_test = pad_zeros(enodedtSet_test)
    testDataSet = create_data_loader(enodedtSetPaddes_test,encodedlabels_test)


    
    num_correct = 0
    #testing for lstm
    for inputs, labels in testDataSet:
        test_output, test_h = net.forward(inputs)
        loss = criterion(test_output, labels.float())
        preds = torch.round(test_output.squeeze())
        correct_tensor = preds.eq(labels.float().view_as(preds))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)
        
    
    print("Test Accuracy: {:.2f}".format(num_correct/len(testDataSet.dataset)))

#################################################################################
######################## GRUMODEL ########################################
#################################################################################

#################################################################################
######################## selfAttn ###############################################
#################################################################################

def runSelfAttention():
    cumulatedList = read_file('./movie_reviews.tar.gz')
    DATA = cumulatedList[0]
    LABELS = cumulatedList[1]
    TOKENDICT = preprocess(DATA)
    enodedtSet = encode_review(TOKENDICT,DATA)
    vocabsize = len(TOKENDICT)
    encodedlabels = encode_labels(LABELS)
    # print(enodedtSet)
    enodedtSetPaddes = pad_zeros(enodedtSet)
    max_length = len(enodedtSetPaddes[0])
    embedding_dict = load_embedding_file('./wiki-news-300d-1M.vec',TOKENDICT)
    trainDataSet = create_data_loader(enodedtSetPaddes,encodedlabels)

    embedding_matrix = torch.zeros((vocabsize+1, 300))
    counter = 0
    for word in TOKENDICT:
        embedding_vector = embedding_dict.get(TOKENDICT[word])
        if embedding_vector is not None:
            embedding_matrix[counter] = embedding_vector
            counter = counter+1
# set hyper_parameters
    print("training self-attention Model")
    vocab_size = vocabsize
    output_size = 1
    learning_rate = 0.01
    n_epochs = 4
    net = AttentionSentiment(50, 1, 16, vocabsize+1,300,embedding_matrix)
    print(net)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #training for lstm
    for epoch in range(1, n_epochs + 1):
            counter = 0
            for inputs, labels in trainDataSet:
                counter += 1
                net.zero_grad()
                prediction = net(inputs)
                loss = criterion(prediction.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                if counter==38:
                    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                    print("Loss: {:.4f}".format(loss.item()))

    print("testing self-attention Model")
    # create test set
    cumulatedList_test = read_file_test('./movie_reviews.tar.gz')
    DATA_test = cumulatedList_test[0]
    LABELS_test = cumulatedList_test[1]
    TOKENDICT_test = preprocess(DATA_test)
    enodedtSet_test = encode_review(TOKENDICT_test,DATA_test)
    vocabsize_test = len(TOKENDICT_test)
    encodedlabels_test = encode_labels(LABELS_test)
    enodedtSetPaddes_test = pad_zeros(enodedtSet_test)
    testDataSet = create_data_loader(enodedtSetPaddes_test,encodedlabels_test)


    
    num_correct = 0

    #testing for lstm
    for inputs, labels in testDataSet:
        test_output = net.forward(inputs)
        loss = criterion(test_output, labels.float())
        preds = torch.round(test_output.squeeze())
        correct_tensor = preds.eq(labels.float().view_as(preds))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)
        
    print("Test Accuracy: {:.2f}".format(num_correct/len(testDataSet.dataset)))

#################################################################################
######################## selfAttn ###############################################
#################################################################################


if __name__ == "__main__":
    print(sys.argv)
    if str(sys.argv[1])=='1':
        runBaseLine()
    if str(sys.argv[1])=='2':
        runLSTM()#pass true if testing for bidirectional
    if str(sys.argv[1])=='3':
        SimpleVanillaRNN()#pass true if testing for bidirectional
    if str(sys.argv[1])=='4':
        GRUMODEL()#pass true if testing for bidirectional
    if str(sys.argv[1])=='5':
        runSelfAttention()




