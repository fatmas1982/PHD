#! /usr/bin/python3.5


# In[2]:

import math

import tensorflow as tf

import csv
from tensorflow.python.client import timeline
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from collections import Counter
import threading

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from string import digits
import os
import math
import sys


# In[3]:

#gpuN="0"#sys.argv[1]
task_no=sys.argv[1]
all_gpus=sys.argv[2]
#cuda="0,1"#sys.argv[3]
processor=sys.argv[4]
dataset=sys.argv[5]
#os.environ['CUDA_VISIBLE_DEVICES'] = cuda


# In[4]:

#dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path ="/home/helwan003u1"
path_data_source=dir_path+"/data/data_source/"#wiki/wikipedia2text/"
#path_data_source="/home/fsg/Desktop/"

path_data_base=dir_path+"/data/database/"+processor+dataset+"/"
#path_data_base=dir_path+"/data/database/csv2/"


#path_data_base="/home/fsg/Desktop/csv-fatma/"

files_path_data_source=dataset+"/"
#files_path_data_source="files/"

#files_path_data_source="mincorpus/"#"corpus/"#"demo/"

#sub_path_data_source="small/"

file_path=path_data_source+files_path_data_source


file_names = [os.path.join(file_path, f) 
                      for f in os.listdir(file_path) 
                      if f.endswith(".txt")]


path_tf="sub_tf/"
path_idf="sub_idf/"
path_tfidf="sub_tfidf/"
path_non_redundant="sub_word_tf/"
path_sim="semantics/sim/"

file_path_tf=path_data_base+path_tf
file_path_idf=path_data_base+path_idf
file_path_tfidf=path_data_base+path_tfidf
file_path_non_redundant=path_data_base+path_non_redundant
file_path_sim=path_data_base+path_sim

file_names_tf = [os.path.join(file_path_tf, f) 
                      for f in os.listdir(file_path_tf) 
                      if f.endswith(".csv")]
file_names_idf = [os.path.join(file_path_idf, f) 
                      for f in os.listdir(file_path_idf) 
                      if f.endswith(".csv")]


file_names_tfidf = [os.path.join(file_path_tfidf, f) 
                      for f in os.listdir(file_path_tfidf) 
                      if f.endswith(".csv")]


file_names_non_redundant = [os.path.join(file_path_non_redundant, f) 
                      for f in os.listdir(file_path_non_redundant) 
                      if f.endswith(".csv")]


file_names_sim = [os.path.join(file_path_sim, f) 
                      for f in os.listdir(file_path_sim) 
                      if f.endswith(".csv")]


# In[5]:

import subprocess, re, os, sys #https://github.com/yaroslavvb/stuff/blob/master/notebook_util.py
def run_command(cmd):
    """Run command, return output as string."""
    
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


# In[6]:

def list_available_gpus():
    """Returns list of available GPU ids."""
    
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result


# In[7]:

def save_txt(txt,file):
    text_file = open(file, "w")
    text_file.write(txt)
    text_file.close()


# In[8]:

def gpu_memory_map(gpu_memory_file,gpu_output_file):
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    save_txt(output,gpu_output_file)
    #print("nvidia-smi",output)
    gpu_output = output[output.find("GPU Memory"):]
    #print("GPU Memory",gpu_output)
    save_txt(gpu_output,gpu_memory_file)
   
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    #print("memory_regex",memory_regex)
    rows = gpu_output.split("\n")
    #print("rows",rows)
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    #print("result",result)
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


# In[ ]:




# In[9]:

def read_cvs_by_pands(path_database,file_database,index_col, header):
    import csv
    import pandas as pd
    df=pd.read_csv(path_database+file_database,index_col=index_col,header=header)
    
    return df#pd.read_csv(path_database+file_database,index_col=index_col,header=header)



# In[1]:


def isfile_empty(file_path_name):
    f=open(file_path_name, 'r',encoding='utf-8') 
    is_blank = len(f.read().strip()) == 0
    return is_blank


# In[3]:

#isfile_empty("/home/fsg/Desktop/cssplit225.csv")


# In[18]:

def write_cvs_by_pands(path_database,file_database,header,data_rows):
    import csv
    import pandas as pd
    csv_df=pd.DataFrame(data_rows,columns=header ) 
    csv_df.to_csv(path_database+file_database)


# In[19]:

def read_sord_tf_file(path_data_source,sub_path_data_source_tf,i):
    #df=read_cvs_by_pands(path_data_source,sub_path_data_source_tf+"cs"+str(i)+".csv",0,header=0)
    #print(path_data_source+sub_path_data_source_tf+i)
    df=read_cvs_by_pands(path_data_source,sub_path_data_source_tf+i,0,header=0)
    #ee.T.sort_index(inplace=True)
    df = df.T.sort_index()
    return df


# In[20]:

def read_sord_idf_file(path_data_source,sub_path_data_source_idf,i):
    #df=read_cvs_by_pands(path_data_source,sub_path_data_source_idf+"cs"+str(i)+".csv",0,header=0)
    df=read_cvs_by_pands(path_data_source,sub_path_data_source_idf+i,0,header=0)
    #ee.T.sort_index(inplace=True)
    df = df.sort_index()
    return df


# In[21]:



'''
Write Excell sheet
'''
def save_file_to_database(data_rows,path_database,file_databbase,header_list):
    import csv
    outfile = open(path_database+file_databbase,'w')
    writer=csv.writer(outfile)
    #header_list=['uuid','paragraph','doc_id']
    i=0
    for line in data_rows:
        row=[i,line,'paragraph no.'+str(i)]
        if i==0:
            
            writer.writerow(header_list)
            writer.writerow(row)
        else:
            ##print('ff')
            writer.writerow(row)
        i+= 1
        #outfile.close()


# In[22]:

'''
Read Excell sheet
'''
def read_text_from_database(path_database,file_databbase):
    import csv
    queue_paragraph=[]
    #f = open(sys.argv[1], 'rt')
    outfile = open(path_database+file_databbase,'rt')
    try:
                
        reader=csv.reader(outfile)
        for row in reader:
            queue_paragraph.append(row)
            ##print (row)
    finally:
        ##print ("row")
        outfile.close()
        
    return queue_paragraph
    


# In[23]:

def add_row_csv(path_database,idf,list_data):
    import csv
    with open(path_database+idf, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(list_data)
        f.close()
        
       


# In[24]:

def create_file(path_data_source,file_name):
    import csv
    outfile = open(path_data_source+file_name,'w')
    writer=csv.writer(outfile)
    outfile.close()


# In[ ]:




# In[25]:

def is_file_exist(file_path,file_name):
    file_names = [os.path.join(file_path, f) 
                      for f in os.listdir(file_path) 
                      if f.endswith(".csv")]
    #print(file_names)
    if file_name in file_names:
        return True
    else:
        return False
    


# In[ ]:




# In[26]:

#save pragraphs to files
def write_file(pragraph,num_pragraph,path):
    file = open(path+str(num_pragraph)+".txt","w") 
 
    file.write(pragraph) 
    
    file.close() 
    


# In[27]:

#create sub dataset
def sub_dataset(path_data_source,data_source):
    pragraphs=txt_pragraphs(read_file(path_data_source+data_source))
    counter=0
    for pragraph in pragraphs:
        ##print('pragraph no ',counter)
        write_file(pragraph,counter,sub_path_data_source)
        counter +=1
    


# In[28]:

def read_file(str):
    file = open(str,'r',encoding='utf-8')
    txt=file.read()
    ##print(txt)
    return txt


# In[29]:

def txt_pragraphs(str):
    pragraphs = str.split("\n\n")
    return pragraphs
#pragraphs=txt_pragraphs(txt)
#type(pragraphs)


# In[30]:

def pragraph_to_setnences(str):
    from nltk.tokenize import sent_tokenize, word_tokenize
    return sent_tokenize(str)
#setnences=pragraph_to_setnences(pragraphs[n_pragraph])


# In[31]:

new_stop_words = ['the', 'that', 'to', 'as', 'there', 'has', 'and', 'or', 'is', 'not', 'a', 'of', 'but', 'in', 'by', 'on', 'are', 'it', 'if','what','where','how','when']
new_stop_words2=['--','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now','even','until','then','must']
numbers=[1,2,3,4,5,6,7,8,9]
#stemmer = SnowballStemmer("english", ignore_stopwords=True)
def remove_stopword_sentences(sent):
    import nltk
    from nltk.corpus import wordnet as wn
    import time
    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer
    from string import digits
    from nltk.corpus.reader.wordnet import WordNetError
    import sys
    list_word=[]

    try:
        tokenizer = RegexpTokenizer("[\w']+")
    
        words=tokenizer.tokenize(sent)
    
        english_stops = set(stopwords.words('english'))
        #stems=[]
        
        list_word=[word for word in words if word.lower() not in english_stops and word.lower() not in new_stop_words and word.lower() not in new_stop_words2 and  not word.lower().isdigit() and word.lower() not in digits and word.lower() not in  numbers]
    
    #for word in list_word:
        #stems.append(stem(word))
        #stems.append(PorterStemmer().stem(word))
        #stems.append(stemmer.stem(word))
        #stems.append(stemmer.stem("computer"))
        #stems.append(word)
    except WordNetError as e:
        print("WordNetError on concept {}: {}".format("remove_stopword_sentences: ",e))
    except AttributeError as e:
        print("Attribute error on concept {}: {}".format("remove_stopword_sentences: ", e))
    except:
        print("Unexpected error on concept {}: {}".format("remove_stopword_sentences: ", sys.exc_info()[0]))
    
    return list_word#stems#(stem(setem_word for setem_word in  ([word for word in words if word not in english_stops and word not in new_stop_words])))


# In[32]:

#remove_stopword_sentences("//jdnf dfd \ dfjfd-eee")


# In[33]:

def word_list_sentece(pragraph):
    words_list=[]
    setnences=pragraph_to_setnences(pragraph)
    for indexs in range(len(setnences)):    
        ##print("Sentence No. ",indexs,": ",setnences[indexs],"\n")
        words=remove_stopword_sentences(setnences[indexs])
        wordsent=''
        for index in range(len(words)):
            wordsent+=' '+words[index]
            ##print("wordsent:",wordsent)
            
        words_list.append(wordsent)
        #count = Counter(words)
        ##print("wordsent:",wordsent)
        ##print(" word:",words)
    ##print(words_list)
    return words_list

#corpus=word_list_sentece(pragraphs[0])


# In[34]:


'''
this function for compute lesk for each word(list of word) in sentence
'''
def lesk_words_sentence(words,sentence):
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk
    lesks= []
    for word in words:
        if lesk(sentence,word, 'n') is not None:
            lesks.append(lesk(sentence,word, 'n'))
            ##print("Word is: ",word,"\n LESK: ",lesk(sentence,word, 'n'),"\n Sentence: ",sentence )
        
    return lesks


# In[35]:

'''
this function for compute lesk of word in sentence
'''

def lesk_word_sentence(sentence,word):
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk
    from nltk.corpus.reader.wordnet import WordNetError
    import sys
    disambiguated=''
    ##print(type(disambiguated))
    try:
        
    #lesks= []
    #for word in words:
    #disambiguated=lesk(context_sentence=sentence, ambiguous_word=word)
    
        disambiguated=lesk(sentence,word, 'n')
        ##print(type(disambiguated))
    ##print(disambiguated)
    #if disambiguated is not None:
        #lesk_synset=disambiguated
    #else:
    #lesk_synset=0
    ##print("Word is: ",word,"\n LESK: ",lesk(sentence,word, 'n'),"\n Sentence: ",sentence )
    except WordNetError as e:
        print("WordNetError on concept {}: {}".format("lesk_word_sentence: ",e))
    except AttributeError as e:
        print("Attribute error on concept {}: {}".format("lesk_word_sentence: ", e))
    except:
        print("Unexpected error on concept {}: {}".format("lesk_word_sentence: ", sys.exc_info()[0]))    
    return disambiguated

#lesk("Computer science is a discipline that spans theory and practice","science")

#sent = 'people should be able to marry a person of their choice'.split()
#lesk(sent, 'able')


# In[ ]:




# In[ ]:




# In[36]:

def idf_df(df,D,base):
    #[7/df['0']]
    y = [log_idf(D,x,base) for x in df['0']]
    return y


# In[37]:

def log_idf(D,d,base):
    return math.log((D/d), base)


# In[38]:

def tf_idf_one(path_data_source,sub_path_data_source_tf,sub_path_data_source_idf,i,sub_path_data_source_tfidf):
    df_tf=read_sord_tf_file(path_data_source,sub_path_data_source_tf,i)
    df_idf=read_sord_idf_file(path_data_source,sub_path_data_source_idf,i)

    if len(df_idf) != 0:
        idf=idf_df(df_idf,len(df_idf),10)

        full_tfidf=[]
        for index in range(len(idf)):
            ##print(index)
            tfidf=df_tf[0][index]*idf[index]
            full_tfidf.append(tfidf)


    df_tf_idf=pd.DataFrame(full_tfidf)
    df_tf_idf.index=df_tf.index
    #df_tf_idf.to_csv(path_data_source+sub_path_data_source_tfidf+"cs"+str(i)+".csv")
    df_tf_idf.to_csv(path_data_source+sub_path_data_source_tfidf+i)
    


# In[ ]:




# In[4]:

def full_name_file(full_name_path):#like cs.csv
    d=full_name_path.split("/")
    ##print(d)
    name=d[len(d)-1]#.split(".")
    return name


# In[1]:

def name_file(full_name_path): #like cs
    d=full_name_path.split("/")
    ##print(d)
    name=d[len(d)-1].split(".")
    return name[0]


# In[5]:

#full_name_file("/data/database/csv/split156.csv")


# In[42]:

'''
merge two dictionary 
d={'a': 1, 'c': 3, 'k': 5}
d1={'g': 1, 'c': 3, 'b': 5}
like merge(d, d1,lambda x,y: x+1)
{'a': 1, 'b': 5, 'c': 4, 'g': 1, 'k': 5}
'''

def merge(d1, d2, merge_fn=lambda x,y:y):
    """
    Merges two dictionaries, non-destructively, combining 
    values on duplicate keys as defined by the optional merge
    function.  The default behavior replaces the values in d1
    with corresponding values in d2.  (There is no other generally
    applicable merge strategy, but often you'll have homogeneous 
    types in your dicts, so specifying a merge technique can be 
    valuable.)

    Examples:

    >>> d1
    {'a': 1, 'c': 3, 'b': 2}
    >>> merge(d1, d1)
    {'a': 1, 'c': 3, 'b': 2}
    >>> merge(d1, d1, lambda x,y: x+y)
    {'a': 2, 'c': 6, 'b': 4}

    """
    result = dict(d1)
    for k,v in d2.items():
        if k in result:
            result[k] = merge_fn(result[k], v)
            ##print(k)
        #else:
            #result[k] = v
    return result


# In[43]:

def match_lists(list_one,list_two):
    dic_one=list_to_dict_one(list_one)
    dic_two=list_to_dict_one(list_two)
    return merge(dic_one, dic_two,lambda x,y: x+1)


# In[44]:

#convert list to dic has value 1
def list_to_dict_one(my_list):
    my_dict = {k: 1 for k in my_list} 
    return my_dict


# In[ ]:




# In[45]:

'''
Write Excell sheet
'''
def save_list_to_csv(data_rows,path_data_base,path_file,file_name):
    import csv
    outfile = open(path_data_base+path_file+file_name,'w')
    writer=csv.writer(outfile)
    
    writer.writerow(data_rows)
     
    outfile.close()
            


# In[ ]:




# In[46]:

def match_file_files(one_file,list_files):
    path_database=dir_path+"/data/database/csv/"
    path_sub_idf="sub_idf/"
    
    word_list_one_file=csv_to_list(one_file)
    ##print(type(word_list_one_file))
    dict_one_word_list=list_to_dict_one(word_list_one_file[0])
    ##print(dict_one_word_list)
   
    for i in range(len(list_files)):  
        
        ##print("***************",i,"******************")
        filename=list_files[i]
        if one_file != filename:
            ##print(name_file(filename))
            word_list=csv_to_list(filename)
            dict_word_list=list_to_dict_one(word_list[0])
            ##print(dict_word_list)
            dict_one_word_list=merge(dict_one_word_list, dict_word_list,lambda dict_one_word_list,dict_word_list:dict_one_word_list+1)
            ##print(dict_one_word_list)
        #else:
            ##print("equal")
    write_cvs_by_pands(path_database+path_sub_idf,name_file(one_file)+'.csv',dict_one_word_list)
    ##print('\n',merg_dict.keys())
    return dict_one_word_list


# # TF Process

# In[47]:

def file_to_LESK_TF(filename,path_database,path_tf): 
    path_database=path_database#dir_path+"/data/database/csv/"
    #path_sub_tfidf=path_sub_tfidf#"sub_word_tf/"
    #path_full_tfidf=path_full_tfidf#"full_word_tf/"
    path_tf=path_tf#"sub_tf/"
    #TF_File="TF-"
    #TF_Full="TF-Full.csv"

    #for i in range(len(file_list_task)):    
    #with tf.Session(config=config) as sess:
    #index_paragraph=0
    col=1

    #index_file=0
    #sess.run(tf.global_variables_initializer())
    ##print(file_names)

    #for filename in file_names:
        ##print("index_file",str(index_file))
    word_file_fatma=[]
    #filename=file_list_task[i]
    with open(filename,encoding='utf-8') as inf:
            ##print("tpe",type(inf))
        txt=inf.read()

        paragraph_list=txt_pragraphs(txt)   


        for paragraph in paragraph_list: #get pragraphs(documents) from DB
                ##print("Pragraph type ",type(paragraph))


            #if index_paragraph ==0:
                #index_paragraph += 1
            #else:

            setnences=pragraph_to_setnences(paragraph)#partitions paragraph to sentence


            for setnence in setnences:
                        ##print("  ",setnence)                            

                words=remove_stopword_sentences(setnence)#remove stop words and noise
                #try:

                for word in words:
                    try:

                            lesk=lesk_word_sentence(setnence,word)#get LESK of word in sentence


                            #paragraph_word.append(word_sentence)

                            if lesk is not None:
                            ##print("type of lesk in words",type(lesk),lesk)

                                word_file_fatma.append(lesk.name())

                    except WordNetError as e:
                            print("WordNetError on concept {}:{}".format("My model "+word+" "+lesk,e))
                    except AttributeError as e:
                            print("Attribute error on concept {}:{}".format("My model "+word+" "+lesk,e))
                    except:
                            print("Unexpected error on concept {}:{}".format("My model "+word+" "+lesk,sys.exc_info()[0]))



                '''////////////////END Sentence////////////////# '''


            #write_cvs_by_pands(path_database,word_sentences_table,word_sentences_list,word_sentences_list_data)


            '''////////////////END PARAGRAPH////////////////# '''

    #write_cvs_by_pands(path_database,sentences_paragraph_table,sentences_paragraph_list,sentences_paragraph_list_data)

    ##print(word_file_fatma)

    word_file_Freq=Counter(word_file_fatma)
    sum_count=sum(word_file_Freq.values())

    ##print(type(word_file_Freq))
    ##print(word_file_Freq)
    #csv_df=pd.DataFrame([word_file_Freq],columns=word_file_Freq.keys() ) 
    freq=[]
    for i in word_file_Freq.values():
        c=i/sum_count
        freq.append(c)
    csv_df=pd.DataFrame([freq],columns=word_file_Freq.keys() ) 

    #Save TF file
    #new_file_name="cs"+name_file(filename)+".csv"
    new_file_name=name_file(filename)+".csv"
    csv_df.to_csv(path_database+path_tf+new_file_name)
        # add to idf file 

    #full_list=[]
    #full_list.insert(0,name_file(filename)) # to add name of file in the firest cell like cs1 or cs4
    #full_list=full_list+list(word_file_Freq.keys())
    # add to single

    #add_row_csv(path_database+path_sub_tfidf,full_name_file(filename),list(word_file_Freq.keys()))
    # add to total idf file 
    #add_row_csv(path_database+path_full_tfidf,TF_Full,list(word_file_Freq.keys()))
    #index_file +=1
    return new_file_name


# In[48]:

#with open("/home/fsg/Desktop/split0.txt",encoding='utf-8') as inf:
    #txt=inf.read()
    #print(txt)


# In[49]:

#"ff/sduy.2.2.01".replace("/", "_")


# # Non Redundant 

# In[50]:

d1={'a': 1, 'b': 1, 'c': 1}
d2={'d': 1, 'x': 1, 'b': 1}


# In[51]:

'''
d1={'a': 1, 'b': 1, 'c': 1}
d2={'d': 1, 'x': 1, 'b': 1}
result={'a': 1, 'c': 1}
'''

def dict_remove_redundant(dic1,dic2):
    
    dic1=merge(dic1, dic2,lambda dic1,dic2:dic1*0)
    #print("\n")
    #print("in remove",len(dic1),dic1,"\n")
    dic1=dict((k,v) for k, v in dic1.items() if v)
    #print("in remove 2",len(dic1),dic1,"\n")
    return dic1

#dict_remove_redundant(d1,d2)


# In[52]:

def dict_key_to_list(dic):#like dict_keys(['a', 'c'])
    
    return dic.keys()


# In[53]:

def read_last_file_list(file_path,extention):
    
    file_names = [os.path.join(file_path, f) 
                      for f in os.listdir(file_path) 
                      if f.endswith(extention) and not isfile_empty(file_path+f)]
    return file_names


# In[ ]:




# In[54]:

def name_file_no(full_name_path): #like cs
    d=full_name_path.split("/")
    ##print(d)
    name=d[len(d)-1].split(".")
    return name[0][5:]
    


# In[55]:

#name_file_no("/data/database/csv/split123.csv")


# In[56]:

def read_last_file_list_sim_previouse(file_path,extention,no_task):
    import os
    file_names = [os.path.join(file_path, f) 
                      for f in os.listdir(file_path) 
                      if f.endswith(extention) and not isfile_empty(file_path+f) and int(name_file_no(f))<int(no_task)]
    return file_names


# In[61]:

#read_last_file_list_sim_previouse('/home/fsg/Desktop/csv/sub_word_tf/',".csv","2")


# In[59]:

def remove_redundant(tf_file,path_data_base,path_tf,path_non_redundant):
    print("remove_redundant",tf_file)
    list_tf=read_cvs_by_pands(path_data_base+path_tf,tf_file,0,0).keys()
    dic_tf=list_to_dict_one(list_tf)
    ##print(dic_tf)
    
    file_names_non_redundant=read_last_file_list(path_data_base+path_non_redundant,".csv")
    ##print("file_names_non_redundant",file_names_non_redundant)
    for file_nonredun in file_names_non_redundant:
        ##print("file_nonredun")
        pure_file_name=full_name_file(file_nonredun)
        ##print(pure_file_name)
        list_non=read_cvs_by_pands(file_path_non_redundant,pure_file_name,None,0)
        
        dic_non=list_to_dict_one(list_non)
        #print("list_non",dic_non)
        dic_tf=dict_remove_redundant(dic_tf,dic_non)#///////////////////
        #print("dic_tf",len(dic_tf),dic_tf)
    list_term=dict_key_to_list(dic_tf)
    #print("list_term",len(list_term))
    
    save_list_to_csv(list_term,path_data_base,path_non_redundant,tf_file)
    


# In[45]:

#remove_redundant("cs1.csv",path_data_base,path_tf,path_non_redundant)
   


# # IDF

# In[46]:

def dict_IDF(dic1,dic2):
    
    dic3=merge(dic1, dic2,lambda dic1,dic2:dic1+1)
    dic4=merge(dic2, dic1,lambda dic2,dic1:dic2+1)
    
    return dic3,dic3


# In[47]:

#d1={'a': 1, 'b': 1, 'c': 1}
def dict_to_DF(dic):
    df=pd.DataFrame([dic])
    return df

#dict_to_DF(d1)


# In[48]:

def save_df_to_csv(df,path_database,sub_path,new_file_name):
     df.to_csv(path_database+sub_path+new_file_name)


# In[49]:

def magic(numList):         # [1,2,3]
    s = map(str, numList)   # ['1','2','3']
    s = ''.join(s)          # '123'
    s = int(s)              # 123
    return s


# In[50]:

def df_to_dict(df):
        
    dic={}
    keys=df.keys()
    
    values= df.T.values.tolist()
    #print(len(values))
    for i in range(len(keys)):
        #print(keys[i])
        dic[keys[i]]=magic(values[i])
    return dic


# In[51]:

#df=read_cvs_by_pands(path_data_base+path_idf,"cs0.csv",0,0)
#df_to_dict(df)


# In[52]:

def Idf(path_data_base,path_tf,tf_file_name,path_idf):
    #tf_file_name="cs2.csv"
    list_tf=read_cvs_by_pands(path_data_base+path_tf,tf_file_name,0,0).keys()

    dic_tf=list_to_dict_one(list_tf)
    #print(dic_tf)

    file_names_IDF=read_last_file_list(path_data_base+path_idf,".csv")


    for file_IDF in file_names_IDF:
            old_dic_tf_updated=dic_tf.copy()
            ##print("file_nonredun")
            pure_file_name=full_name_file(file_IDF)
            #print("pure_file_name")
            #open this file name as list with value
            df_idf=read_cvs_by_pands(path_data_base+path_idf,pure_file_name,0,0)#.keys()
            #print("df_idf \n")
            #print(df_idf )
            #convert list to dic dict_idf
            dict_idf=df_to_dict(df_idf)
            #merge dic_tf with dict_idf
            dic_tf=merge(dic_tf, dict_idf,lambda dic_tf,dict_idf:dic_tf+1)
            #if dic_tf changed 
            if old_dic_tf_updated != dic_tf:
                #print("yeeeeeeeees")
                #merge dict_idf  with dic_tf 
                dict_idf=merge(dict_idf, dic_tf,lambda dict_idf,dic_tf:dict_idf+1)
                #convert dict_idf to df_idf
                df_idf_updated=dict_to_DF(dict_idf)
                #save df_idf to csv
                save_df_to_csv(df_idf_updated,path_data_base,path_idf,pure_file_name)

    df_idf=dict_to_DF(dic_tf)
    save_df_to_csv(df_idf,path_data_base,path_idf,tf_file_name)



# In[53]:

def sub_list_file(file_list_task,all_gpus):
    import math
    sub_len=math.ceil(len(file_list_task)/all_gpus)
    global_list_len=math.ceil(len(file_list_task)/sub_len)
    
    global_list=[]
    index=0
    for x in range(global_list_len):
        sublist=[]
        for i in range(sub_len):
            if index < len(file_list_task):
                sublist.append(file_list_task[index])
                index +=1
                
        global_list.append(sublist)

    return global_list
    


# # Semantic

# In[3]:

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
from nltk.tokenize import word_tokenize

def similarity_by_infocontent(sense1, sense2, option):
    #sense1="Synset('"+sense1+"')"
    #sense2="Synset('"+sense2+"')"
    #print(sense1,sense2)
    sense1 = wn.synset(sense1)
    sense2 = wn.synset(sense2)
    #print(sense1,sense2)
    """ Returns similarity scores by information content. """
    #if sense1.pos != sense2.pos: # infocontent sim can't do diff POS.
        #return 0

    info_contents = ['ic-bnc-add1.dat', 'ic-bnc-resnik-add1.dat', 
                     'ic-bnc-resnik.dat', 'ic-bnc.dat', 

                     'ic-brown-add1.dat', 'ic-brown-resnik-add1.dat', 
                     'ic-brown-resnik.dat', 'ic-brown.dat', 

                     'ic-semcor-add1.dat', 'ic-semcor.dat',

                     'ic-semcorraw-add1.dat', 'ic-semcorraw-resnik-add1.dat', 
                     'ic-semcorraw-resnik.dat', 'ic-semcorraw.dat', 

                     'ic-shaks-add1.dat', 'ic-shaks-resnik.dat', 
                     'ic-shaks-resnink-add1.dat', 'ic-shaks.dat', 

                     'ic-treebank-add1.dat', 'ic-treebank-resnik-add1.dat', 
                     'ic-treebank-resnik.dat', 'ic-treebank.dat']

    if option in ['res', 'resnik']:
        #return wn.res_similarity(sense1, sense2, wnic.ic('ic-bnc-resnik-add1.dat'))
        #print('simRe snik (c1,c2) = -log p(lso(c1,c2)) = IC(lso(c1,c2)')
        return wn.res_similarity(sense1, sense2, wnic.ic('ic-treebank-resnik-add1.dat'))
    #return min(wn.res_similarity(sense1, sense2, wnic.ic(ic)) \
    #             for ic in info_contents)

    elif option in ['jcn', "jiang-conrath"]:
        #return wn.jcn_similarity(sense1, sense2, wnic.ic('ic-bnc-add1.dat'))
        #print('sim(jcn) (c1,c2 )= (IC(c1) + IC(c2 )) - 2IC(lso(c1,c2 ))')
        return wn.jcn_similarity(sense1, sense2, wnic.ic('ic-treebank.dat'))

    elif option in ['lin']:
        #return wn.lin_similarity(sense1, sense2, wnic.ic('ic-bnc-add1.dat'))
        #print('sim(lin) (c1,c2)=(2IC(lso(c1,c2 )))/(IC(c1)+IC(c2))')
        return wn.lin_similarity(sense1, sense2, wnic.ic('ic-treebank.dat'))

def sim(sense1, sense2, option="path"):
    """ Calculates similarity based on user's choice. """
    option = option.lower()
    if option.lower() in ["path", "path_similarity", 
                        "wup", "wupa", "wu-palmer", "wu-palmer",
                        'lch', "leacock-chordorow"]:
        return similarity_by_path(sense1, sense2, option) 
    elif option.lower() in ["res", "resnik",
                          "jcn","jiang-conrath",
                          "lin"]:
        return similarity_by_infocontent(sense1, sense2, option)

def max_similarity(context_sentence, ambiguous_word, option="path", 
                   pos=None, best=True):
    """
    Perform WSD by maximizing the sum of maximum similarity between possible 
    synsets of all words in the context sentence and the possible synsets of the 
    ambiguous words (see http://goo.gl/XMq2BI):
    {argmax}_{synset(a)}(\sum_{i}^{n}{{max}_{synset(i)}(sim(i,a))}
    """
    result = {}
    for i in wn.synsets(ambiguous_word):
        try:
            if pos and pos != str(i.pos()):
                continue
        except:
            if pos and pos != str(i.pos):
                continue
        result[i] = sum(max([sim(i,k,option) for k in wn.synsets(j)]+[0])                         for j in word_tokenize(context_sentence))

    if option in ["res","resnik"]: # lower score = more similar
        result = sorted([(v,k) for k,v in result.items()])
    else: # higher score = more similar
        result = sorted([(v,k) for k,v in result.items()],reverse=True)
    #print (result)
    if best: return result[0][1];
    return result


# In[5]:

#similarity_by_infocontent('read/write_head.n.01', 'read/write_head.n.01', 'res')


# In[56]:

def sim_terms_one_file(path_data_base,path_non_redundant,file_name):
    #print("Start_sim",path_data_base,path_non_redundant,file_name,file_path_sim)
    #list_terms=read_cvs_by_pands(path_data_source,sub_path_data_source_tfidf+file_name,0,0)#.index
    list_terms=list(read_cvs_by_pands(path_data_base,path_non_redundant+file_name,None,0))
    #compare between the same file
    #print(len(list_terms))
    #index=1
    #limt=5
    for i in range(len(list_terms)):
        
        term=list_terms[i]
        term_file=term.replace("/", "_")
        #print(term)
        is_term_new=False
        if not is_file_exist(file_path_sim,file_path_sim+term_file+".csv"): # if !Fale this new term
                #print(" not term",term)
                #create_file(file_path_sim,term+".csv")  
                is_term_new=True

        for term_next in list_terms[i:]:
            #print("terrm",term,"term_next",term_next)

            is_term_next_new=False

            term_next_file=term_next.replace("/", "_")

            if not is_file_exist(file_path_sim,file_path_sim+term_next_file+".csv"): #next term is new
                    #print(" not termin",list_terms[index])
                    #create_file(file_path_sim,term_next+".csv")
                    is_term_next_new=True

            #print(term,is_term_new,term_next,is_term_next_new)
            sim=0
            list_term=[]
            list_term_next=[]
            if is_term_new or is_term_next_new:
                sim=similarity_by_infocontent(term, term_next, 'res')
                #print(term,term_next,"sim",sim)
                if sim <1:
                    sim=0
                if sim !=0:
                    list_term=[term,sim]
                    list_term_next=[term_next,sim]

            #print(list_term)
            #print(list_term_next)
            if sim !=0:
                if term != term_next:
                    #print("term != term_next")
                    if is_term_new:  
                        #print("             is_term_new",term)
                        #print("is_term_new",file_path_sim+term_file+".csv")
                        create_file(file_path_sim,term_file+".csv")
                        add_row_csv(file_path_sim,term_file+".csv",list_term)#add the same sim
                        add_row_csv(file_path_sim,term_file+".csv",list_term_next)#add the next sim
                        is_term_new=False
                    else:
                        #print("             is_term_old",term)
                        add_row_csv(file_path_sim,term_file+".csv",list_term_next)#add the next sim


                    if is_term_next_new:
                        #print("             is_term_next_new",term_next)

                        sim_nex=similarity_by_infocontent(term_next, term_next, 'res')
                        #print("is_term_next_new ",file_path_sim," term_next",term_next_file+".csv")
                        create_file(file_path_sim,term_next_file+".csv")
                        list_next=[term_next,sim_nex]
                        add_row_csv(file_path_sim,term_next_file+".csv",list_next)#add term to next
                        add_row_csv(file_path_sim,term_next_file+".csv",list_term)#add term to next
                        is_term_next_new=False


                    else:
                        #print("              is_term_next_old",term_next,sim)
                        add_row_csv(file_path_sim,term_next_file+".csv",list_term)#add term to next
                else:
                    #print("term == term_next",term,term_next,sim,list_term)
                    if is_term_new:  
                        #print("            is_term_new",term,sim,list_term)
                        #print("else",file_path_sim+term_file+".csv")
                        create_file(file_path_sim,term_file+".csv")
                        add_row_csv(file_path_sim,term_file+".csv",list_term)#add the same sim
                        is_term_new=False


        #print("finesed")

            
            


# In[57]:

#list_terms=list(read_cvs_by_pands(path_data_base,path_non_redundant+"cscs5.csv",None,0))
#list_terms=list(read_cvs_by_pands(path_data_base,path_non_redundant+"cscs5.csv",None,0))
#list_terms


# In[58]:

#sim_terms_one_file(path_data_base,path_non_redundant,"cs0.csv")


name_file_no


# In[59]:

def sim_terms_previous_file(path_data_base,path_non_redundant,file_name):
    #print("sim_terms_previous_file",path_data_base,path_non_redundant,file_name,file_path_sim)
    
    #previous_file_names_list=read_last_file_list(path_data_base+path_non_redundant,".csv")
    previous_file_names_list=read_last_file_list_sim_previouse(path_data_base+path_non_redundant,".csv",task_no)
    #print("previous_file_names_list",previous_file_names_list)
    #list_terms=read_cvs_by_pands(path_data_source,sub_path_data_source_tfidf+file_name,0,0)#.index
    list_terms=list(read_cvs_by_pands(path_data_base,path_non_redundant+file_name,None,0))
    for previous_file_name in previous_file_names_list:
        #print(previous_file_name)
        pure_file_name=full_name_file(previous_file_name)
        if pure_file_name !=file_name:
            #print(pure_file_name)
            list_terms_others=list(read_cvs_by_pands(path_data_base,path_non_redundant+pure_file_name,None,0))

            #compare between the same file
            #print(len(list_terms))
            #index=1
            limt=5
            for i in range(len(list_terms)):
                term=list_terms[i]
                term_file=term.replace("/", "_")
                #print(term)
                is_term_new=False
                if not is_file_exist(file_path_sim,file_path_sim+term_file+".csv"): # if !Fale this new term
                        #print(" not term",term)
                        #create_file(file_path_sim,term+".csv")  
                        is_term_new=True

                #for term_next in list_terms[i:limt]:
                for x in range(len(list_terms_others)):
                    term_next=list_terms_others[x]
                    term_next_file=term_next.replace("/", "_")
                    #print("terrm",term,"term_next",term_next)

                    is_term_next_new=False



                    if not is_file_exist(file_path_sim,file_path_sim+term_next_file+".csv"): #next term is new
                            #print(" not termin",list_terms[index])
                            #create_file(file_path_sim,term_next+".csv")
                            is_term_next_new=True

                    #print(term,is_term_new,term_next,is_term_next_new)
                    sim=0
                    list_term=[]
                    list_term_next=[]
                    if is_term_new or is_term_next_new:
                        sim=similarity_by_infocontent(term, term_next, 'res')
                        #print(term,term_next,"sim",sim)
                        if sim <1:
                            sim=0
                        if sim !=0:
                            list_term=[term,sim]
                            list_term_next=[term_next,sim]

                    #print(list_term)
                    #print(list_term_next)
                    if sim !=0:
                        if term != term_next:
                            #print("term != term_next")
                            if is_term_new:  
                                #print("             is_term_new",term)
                                #print("is_term_new",file_path_sim+term_file+".csv")
                                create_file(file_path_sim,term_file+".csv")
                                add_row_csv(file_path_sim,term_file+".csv",list_term)#add the same sim
                                add_row_csv(file_path_sim,term_file+".csv",list_term_next)#add the next sim
                                is_term_new=False
                            else:
                                #print("             is_term_old",term)
                                add_row_csv(file_path_sim,term_file+".csv",list_term_next)#add the next sim


                            if is_term_next_new:
                                #print("             is_term_next_new",term_next)

                                sim_nex=similarity_by_infocontent(term_next, term_next, 'res')
                                #print("is_term_next_new",file_path_sim+term_next_file+".csv")
                                create_file(file_path_sim,term_next_file+".csv")
                                list_next=[term_next,sim_nex]
                                add_row_csv(file_path_sim,term_next_file+".csv",list_next)#add term to next
                                add_row_csv(file_path_sim,term_next_file+".csv",list_term)#add term to next
                                is_term_next_new=False


                            else:
                                #print("              is_term_next_old",term_next,sim)
                                add_row_csv(file_path_sim,term_next_file+".csv",list_term)#add term to next
                        else:
                            #print("term == term_next",term,term_next,sim,list_term)
                            if is_term_new:  
                                #print("            is_term_new",term,sim,list_term)
                                #print("else",file_path_sim+term_file+".csv")
                                create_file(file_path_sim,term_file+".csv")
                                add_row_csv(file_path_sim,term_file+".csv",list_term)#add the same sim
                                is_term_new=False


                #print("finesed")


# In[ ]:




# In[60]:

#load list word of current sub_word
#start sim between word and next word
# load next file from previous list 
##start sim between words in curent file and word in next file 
#store each comparison in files


# In[61]:


def gpu_full_process(filename,path_data_base,path_tf,path_non_redundant,path_idf,file_path_sim):
    #print("In gpu file_path_sim",file_path_sim)
    #index_file=0
    #file_list_task=read_last_file_list(path_data_source+files_path_data_source,".txt")
    #for i in range(len(file_list_task)):
    #filename=file_list_task[i]
    #print(filename)
    
    if not isfile_empty(path_data_base+path_non_redundant+filename):
        sim_terms_one_file(path_data_base,path_non_redundant,filename)
        print("finesed sim",filename)
        #sim_terms_previous_file(path_data_base,path_non_redundant,filename)
        #print("finesed sim_terms_previous_file")
    else:
        print("Empty redundant",filename)

      


# In[62]:

#isfile_empty("/home/fsg/Desktop/files/dd.txt")
    


# In[63]:

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    #return [x.name for x in local_device_protos]


# In[14]:

def distrupited_task_gpu(no_task,total_no_gpu):
    
    return no_task%total_no_gpu


# In[65]:

config = tf.ConfigProto(device_count={processor.upper():int(all_gpus)},
                        allow_soft_placement=True,
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1,
                        use_per_session_threads=True,
                        log_device_placement=True)
config.gpu_options.allow_growth = True


import os

import tensorflow as tf
#import threading
import timeit
from tensorflow.python.client import timeline
#start_gpu_memory=gpu_memory_map(gpuN+"gpu_befor_session_memory.txt",gpuN+"gpu_befor_session_out.txt")

#save_txt(str(start_gpu_memory),gpuN+"gpu_befor_session_memory_map.txt")

#print("gpu_memory_map",start_gpu_memory)

with tf.Session(config=config) as sess:
    print("************Started Session Main Process*************")
    sess.run(tf.global_variables_initializer())
    #start_gpu_memory=gpu_memory_map(gpuN+"gpu_after_start_session_memory.txt",gpuN+"gpu_after_start_session_out.txt")
    gpu_name='/'+processor+':0'    
    #print("gpu_memory_map_after_session",start_gpu_memory)       
    with tf.device(gpu_name):#'/cpu'):
        print("************Started Session CPU *************")
        #start = timeit.default_timer()
        #file_list_task=read_last_file_list(path_data_source+path_non_redundant,".csv")
        
        #print("file_list_task",file_list_task)
        #sub_file_list_task=sub_list_file(file_list_task,int(all_gpus))[int(gpuN)]
        #print("sub_file_list_task",sub_file_list_task)
        gpu_no=distrupited_task_gpu(int(task_no),int(all_gpus))
        #gpu_name='/gpu:'+str(gpu_no)

        #with tf.device(gpu_name):
        print("************Started Session GPU *************")
        #print("sublist",sub_file_list_task,"gpu_name",gpu_name)
        #print("task_no",task_no,"gpu_name",gpu_name)
        #print("In session",file_path_sim)
        filename=dataset+task_no+".csv" #"cs"+task_no+".txt" 
        #totals_filename=file_path+filename

        gpu_full_process(filename,path_data_base,path_tf,path_non_redundant,path_idf,file_path_sim)
        #Your statements here

        #stop = timeit.default_timer()

        #print ("tttttttttttttt",stop - start )



# In[66]:

#tf.summary.FileWriter("/home/fsg/logs_tf", g).close()


# In[9]:




# In[13]:




# In[ ]:




# In[ ]:



