import numpy as np
import os
import nltk
import itertools
import io
nltk.download('punkt')
import csv
import pandas as pd

from collections import defaultdict
import json

##TRAINING SET
train_pos = pd.read_csv("./data/trainpos.csv", header= None)
train_pos = train_pos.drop(train_pos.columns[0], axis= "columns")
train_pos= np.array(train_pos)

train_neg = pd.read_csv("./data/trainneg.csv", header= None, )
train_neg = train_neg.drop(train_neg.columns[0], axis= "columns")
train_neg= np.array(train_neg)

Trainset= []
##First 280,000 are positive
for i in range (0,len(train_pos)):
  Trainset.append(train_pos[i][0])

##Last 280,000 are negative
for i in range (0,len(train_neg)):
  Trainset.append(train_neg[i][0])

##TESTING SET
test_pos = pd.read_csv("./data/testpos.csv", header= None)
test_pos = test_pos.drop(test_pos.columns[0], axis= "columns")
test_pos= np.array(test_pos)

test_neg = pd.read_csv("./data/testneg.csv", header= None, )
test_neg = test_neg.drop(test_neg.columns[0], axis= "columns")
test_neg= np.array(test_neg)

Testset= []
##First 19,000 are positive
for i in range (0,len(test_pos)):
  Testset.append(test_pos[i][0])

##Last 19,000 are negative
for i in range (0,len(test_neg)):
  Testset.append(test_neg[i][0])

"""# **Dictionary**"""

#Tokenization
count= 0
x_train = []
for i in range (0,len(Trainset)):
  line= nltk.word_tokenize(Trainset[i])
  line = [w.lower() for w in line]
  x_train.append(line)
  count += 1

# number of tokens per review
no_of_tokens = []
for tokens in x_train:
  no_of_tokens.append(len(tokens))
no_of_tokens = np.asarray(no_of_tokens)

### word_to_id and id_to_word. associate an id to every unique token in the training data
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

##Dictionary size = 675,681

# ##TEST OF DICTIONARY
# Word= x_train[0][3] ##FIRST WORD OF THE TRAINING SET##
# ID_word= word_to_id[Word]
# word_ID= id_to_word[ID_word]

# print ( Word,ID_word, word_ID)
# if Word==word_ID:
#   print ('Passed the test')
# else:
#     print ( "GET your shit fixed")


##TOKENIZATION OF TRAINING SET - FIRST TRIAL
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)

##COUNTING TOKENS (TIMES A WORD APPEARS)
for x in x_train_token_ids:
  for token in x:
    count[token] += 1

# ##TEST OF TOKENIZATION IN TRAINING SET
# word= x_train[11][4]
# ID= x_train_token_ids[11][4]
# word_ID= id_to_word[ID]

# print ( word, ID,word_ID)
# if word==word_ID:
#   print ('Passed the test')
# else:
#     print ( "GET your shit fixed")


##SORTED DICTIONARY FROM MAX TO MIN
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]

## NEW WORD_TO_ID BASED ON SORTED LIST
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}

len(word_to_id)

len(id_to_word)

"""# **Tokenization of Data**"""

## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token

#Training set
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]

#Testing set
x_test = []
for i in range (0,len(Testset)):
  line= nltk.word_tokenize(Testset[i])
  line = [w.lower() for w in line]
  x_test.append(line)

x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

##TEST OF TOKENIZATION - TRAINSET WITH SORTED DICTIONARY
line= 4
item= 0
word= x_train[line][item]
ID= x_train_token_ids[line][item]
word_ID= id_to_word[ID-1]

print ( word, ID,word_ID)
if word==word_ID:
  print ('Passed the test')
else:
    print ( "GET your shit fixed")

##TEST OF TOKENIZATION - TESTSET WITH SORTED DICTIONARY
line= 4
item= 0
word= x_test[line][item]
ID= x_test_token_ids[line][item]
word_ID= id_to_word[ID-1]

print ( word, ID,word_ID)
if word==word_ID:
  print ('Passed the test')
else:
    print ( "GET your shit fixed")

print ('this is dictionary len', len(id_to_word))

"""# **Saving: Tokenized Data and Dictionary**"""

output_directory = './processed_data'
if not os.path.exists(output_directory):
  os.makedirs(output_directory)

# ## save dictionary
vocab_dict_json = defaultdict(int)

for i in range(len(id_to_word)):
  vocab_dict_json[i] = id_to_word[i]

with open("{}/yelp_dictionary.json".format(output_directory), "w") as outfile:
  json.dump(vocab_dict_json, outfile)


# np.save('/content/drive/My Drive/CS547Dataset/yelp_dictionary3.npy',np.asarray(id_to_word))


## SAVE TRAINING DATA TO SINGLE TEXT FILE
with io.open("{}/yelp_training.txt".format(output_directory),'w',encoding='utf-8') as f:
  for tokens in x_train_token_ids:
    for token in tokens:
      f.write("%i " % token)
    f.write("\n")

## SAVE TESTING DATA TO SINGLE TEXT FILE
with io.open("{}/yelp_testing.txt".format(output_directory),'w',encoding='utf-8') as f:
  for tokens in x_test_token_ids:
    for token in tokens:
      f.write("%i " % token)
    f.write("\n")
