'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import io
from pdb import set_trace
import math
import json

from LSTM_model import RNN_model


############################ Loading Dictionary and Vocabulary Length ##############
# Yelp_dictionary = np.load('/media/storage2/jdnunez/Deep_Learning/FinalProject/preprocess_data_juan/yelp_dictionary1.npy')
# word_dict = {n: i for i, n in enumerate(Yelp_dictionary)}
with open ("./processed_data/yelp_dictionary.json", "r") as outfile:
    word_dict = json.load(outfile)

vocab_size = len(word_dict) # number of class(=number of vocab)

############################ Loading and Extracting Data ############################
train_size = 560000
test_size = 38000

# Loading Training Data
x_train = []
with io.open('./processed_data/yelp_training.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:train_size]
y_train = np.zeros((train_size,))
y_train[0:280000] = 1

print("training data loaded successfully...")

# Loading Testing Data
x_test = []
with io.open('./processed_data/yelp_testing.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
x_test = x_test[0:test_size]
y_test = np.zeros((test_size,))
y_test[0:19000] = 1

print("testing data loaded successfully...")

############################ Define the model and Hyperparameters ############################

# Model Hyperparameters
vocab_size += 1
embed_size = 500
n_hidden = 100
n_layer = 1
bidir = False # controls use of bidirectional LSTM
attn = False # controls use of attenction
# n_dir = 2 # 1 for non Bidirectional LSTM, 2 for Bidirectional LSTM

# Training Hyperparameters
batch_size = 100
no_of_epochs = 10 #1000
L_Y_train = len(y_train)
L_Y_test = len(y_test)
train_sequence_len = 100
test_sequence_len = 100

# Model Creation
device = torch.device("cuda: {}".format(0))
model = RNN_model(vocab_size, embed_size, n_hidden, n_layer, device, bidir, batch_size, attention=attn)
model.to(device)

# Criterion and Optimizer to use
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



############################ Main Training & Testing Algorithm ############################
train_loss = []
train_accu = []
test_accu = []
Plot_AccuracyTrain=[]
Plot_AccuracyTest=[]
Plot_RMSE=[]

print("starting training ..")

for epoch in range(no_of_epochs):

    ##################### Training the Model
    model.train()

    epoch_acc_train = 0.0
    epoch_loss_train = 0.0

    epoch_counter = 0

    #time1 = time.time()

    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        x_input = np.zeros((batch_size, train_sequence_len),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < train_sequence_len):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-train_sequence_len+1)
                x_input[j,:] = x[start_index:(start_index+train_sequence_len)]

        y_input = y_train[I_permutation[i:i+batch_size]]

        data = torch.LongTensor(x_input).to(device)#cuda()
        target = torch.LongTensor(y_input).to(device)#cuda()
        optimizer.zero_grad()

        output = model(data,target)

        loss = criterion(output[:,-1,:], target)
        loss.backward()

        optimizer.step()   # update weights

        prediction = model.prob(output[:,-1,:])
        prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        accuracy = len(np.where(prediction==target.cpu().numpy())[0])
        epoch_acc_train += accuracy
        epoch_loss_train += loss.data.item()
        epoch_counter += batch_size

    epoch_acc_train /= epoch_counter
    epoch_loss_train /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss_train)
    train_accu.append(epoch_acc_train)
    Plot_AccuracyTrain.append(epoch_acc_train*100.0)


    print("Training ||", "Epoch:", epoch, "||", "Train Accuracy:","%.2f" % (epoch_acc_train*100.0), "||", "Train Loss:","%.4f" % epoch_loss_train)#, "||", "Epoch Duration:","%.4f" % float(time.time()-time1))


    ##################### Testing the Model
    model.eval()
    RMSE = 0.0

    epoch_acc_test = 0.0
    epoch_loss_test = 0.0

    epoch_counter = 0

    #time1 = time.time()

    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
        x_input = np.zeros((batch_size,test_sequence_len),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < test_sequence_len):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-test_sequence_len+1)
                x_input[j,:] = x[start_index:(start_index+test_sequence_len)]
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = torch.LongTensor(x_input).to(device)#cuda()
        target = torch.LongTensor(y_input).to(device)#cuda()

        with torch.no_grad():
            output = model(data,target)#,train=False)
            loss = criterion(output[:,-1,:], target)

        prediction = model.prob(output[:,-1,:])
        prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        accuracy=len(np.where(prediction==target.cpu().numpy())[0])
        #RMSE += torch.sqrt(torch.FloatTensor(np.asarray(sum((prediction-target)^2)/len(prediction))))
        # RMSE += math.sqrt(sum((prediction-target)^2)/len(prediction))
        epoch_acc_test += accuracy
        epoch_loss_test += loss.data.item()
        epoch_counter += batch_size

    # RMSE /= L_Y_test
    epoch_acc_test /= epoch_counter
    epoch_loss_test /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc_test)
    Plot_AccuracyTest.append(epoch_acc_test*100.0)
    # Plot_RMSE.append(RMSE)

    #time2 = time.time()
    #time_elapsed = time2 - time1

    print("                       ", "Test Accuracy:", "%.2f" % (epoch_acc_test*100.0), "||", "Test Loss", "%.4f" % epoch_loss_test, "||", "Test RMSE", "%.4f" % RMSE)

#torch.save(model,'/media/storage2/jdnunez/Deep_Learning/HW5/Results/2a/overfitted_rnn.model')
#data = [train_loss,train_accu,test_accu,Plot_RMSE]
#data = np.asarray(data)
#np.save('/media/storage2/jdnunez/Deep_Learning/HW5/Results/2a/overfitted_data.npy',data)


#####################################PLOT AFTER ALL iterations (When the loop is done)
Axe_Epochs= list(range(0, num_epochs))

# plotting the line 1 points
plt.plot( Axe_Epochs, Plot_AccuracyTrain, label = "LSTM_TestSet")

# plotting the line 2 points
plt.plot(Axe_Epochs,Plot_AccuracyTest, label = "LSTM_TrainingSet")

# naming the x axis
plt.xlabel('Num. Epochs')
# naming the y axis
plt.ylabel('Accuracy')

# giving a title to my graph
plt.title('LSTM Net')

# show a legend on the plot
plt.legend()

# function to show the plot
print (plt.show() )
