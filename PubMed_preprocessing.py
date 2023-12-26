import pandas as pd
import re
import nltk
import copy
import time

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

'''
Download docword.txt and vocab.txt from https://archive.ics.uci.edu/dataset/164/bag+of+words
'''

numdocs = 300000

def contains_special_characters(string):
    pattern = r'[-!@#$%^&*(),.?":{}|<>+/_]'
    match = re.search(pattern, string)
    return bool(match)

_stopwords = stopwords.words('english')
stemmer = EnglishStemmer()
vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}

with open('vocab.txt', encoding="utf8") as fl:
    vocab = fl.read().split('\n')

word_ids = [idx+1 for idx in range(len(vocab))]
orig_len = len(word_ids)


allstart = time.time()

# First, we are going to filter out invalid words which contain no alphabet, contain digits, special symbols, are too short, have no vowel, etc.
rejection_list = []
last_stem = None
for wid in word_ids:
    word = vocab[wid-1]
    pos = nltk.pos_tag([word])

    if pos[0][1] not in ['NN', 'NNS', 'NNP', 'NNPS']:
      notnoun = 1
    else:
      notnoun = 0
    nochar = int(not any(char.isalpha() for char in word)) #the word contains no character..it is just symbols or numbers
    csc = contains_special_characters(word)
    incomplete = int(len(word) <= 4)
    is_stopword = int(word in _stopwords)
    nothing = int(word == '')
    hasnum = int(any(char.isdigit() for char in word))
    novowel = int(not any(char in vowels for char in word))
    
    if notnoun or nochar or csc or incomplete or is_stopword or nothing or hasnum or novowel:
        rejection_list.append(wid)
word_ids = list(set(word_ids) - set(rejection_list)) #filtering


# Next, we are going to filter out words that appear infrequently
word_prominence = {key: 0 for key in word_ids}
prominence_threshold = 0.045 * numdocs #words that appear in more than 4.5% of the docs will be kept
print('prominence threshold: ', prominence_threshold)

cnt = 0
with open('docword.txt') as fl:
    for line in fl:
        cnt+=1
        if cnt > 3:
            d = line.rstrip()
            d = d.split()
            if d == []: break
            
            docid = int(d[0])
            wordid = int(d[1])
 
            if docid > numdocs: break
            if wordid not in word_ids: continue #already filtered word
            word_prominence[wordid] += 1

rejection_list = []
for wid in word_ids:
    if word_prominence[wid] < prominence_threshold:
      rejection_list.append(wid)
word_ids = list(set(word_ids) - set(rejection_list)) #filtering

new_vocab = []
for wid in word_ids:
    new_vocab.append(vocab[wid-1])
vocab = copy.copy(new_vocab)

print(orig_len, len(word_ids), '\n')

allend = time.time()
print('Total word filteration time:', allend - allstart)


''''''''''''''' STARTING DATA EXTRACTION INTO DATA TABLES '''''''''''''''''''''
allstart = time.time()

chunk_size = 100
cnt = 0
prev_docid = None
first_save = True

df = pd.DataFrame(0, index = range(chunk_size), columns=['document_id'] + word_ids)
idx = -1 #set

with open('docword.txt') as fl:
    for line in fl:
        cnt+=1
        if cnt > 3:
            d = line.rstrip()
            d = d.split()
            if d == []: break
            
            docid = int(d[0])
            wordid = int(d[1])
            wordcount = int(d[2])
            #print(docid)
            if docid > numdocs: break
            if wordid not in word_ids: continue
                  
            if docid != prev_docid: #add new row
                
                if (idx+1) % chunk_size == 0 and idx != -1: #save chunk and create new chunk df
                    #print(docid)
    
                    if first_save:
                      df.to_csv('data.csv', index=False) #write chunk
                      first_save = False
                    else:
                      df.to_csv('data.csv', mode='a', index=False, header=False) #append chunk
                    
                    df = pd.DataFrame(0, index = range(chunk_size), columns=['document_id'] + word_ids)
                    idx = -1 #reset
                
                idx += 1
                prev_docid = docid
                df.loc[idx, 'document_id'] = docid
            
            df.loc[idx, wordid] = wordcount 
               
allend = time.time()
print('Total data table creation time:', allend - allstart)