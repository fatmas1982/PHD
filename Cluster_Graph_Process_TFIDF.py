#! /usr/bin/python3.5


# In[34]:

import math

import tensorflow as tf

import csv
from tensorflow.python.client import timeline
import pandas as pd
import math
import sys
import os


# In[ ]:

task_no=sys.argv[1]
all_gpus =sys.argv[2]
cuda=sys.argv[3]
processor=sys.argv[4]
dataset=sys.argv[5]
#os.environ['CUDA_VISIBLE_DEVICES'] = cuda


# In[35]:


#dir_path = os.path.dirname(os.path.realpath(__file__))

#path_data_source=dir_path+"/data/data_source/"
#path_data_source="/home/fsg/Desktop/"

#path_data_base=dir_path+"/data/database/csv/"


#path_data_base="/home/fsg/Desktop/csv/"


#files_path_data_source="files/"

#files_path_data_source="demo/"

#sub_path_data_source="small/"

dir_path ="/home/helwan003u1"

path_data_source=dir_path+"/data/data_source/"#wiki/wikipedia2text/"

path_data_base=dir_path+"/data/database/"+processor+dataset+"/"

files_path_data_source=dataset+"/"


file_path=path_data_source+files_path_data_source

file_names = [os.path.join(file_path, f) 
                      for f in os.listdir(file_path) 
                      if f.endswith(".txt")]


path_tf="sub_tf/"
path_idf="sub_idf/"
path_tfidf="sub_tfidf/"
path_sim_permutation="semantics/permutation/"
path_sim="semantics/sim/"
path_topic_document="topic_document/"

file_path_tf=path_data_base+path_tf
file_path_idf=path_data_base+path_idf
file_path_tfidf=path_data_base+path_tfidf
file_path_sim_permutation=path_data_base+path_sim_permutation
file_path_sim=path_data_base+path_sim
file_path_topic_document=path_data_base+path_topic_document

file_names_tf = [os.path.join(file_path_tf, f) 
                      for f in os.listdir(file_path_tf) 
                      if f.endswith(".csv")]
file_names_idf = [os.path.join(file_path_idf, f) 
                      for f in os.listdir(file_path_idf) 
                      if f.endswith(".csv")]


file_names_tfidf = [os.path.join(file_path_tfidf, f) 
                      for f in os.listdir(file_path_tfidf) 
                      if f.endswith(".csv")]


file_names_permutation = [os.path.join(file_path_sim_permutation, f) 
                      for f in os.listdir(file_path_sim_permutation) 
                      if f.endswith(".csv")]


file_names_sim = [os.path.join(file_path_sim, f) 
                      for f in os.listdir(file_path_sim) 
                      if f.endswith(".csv")]

file_names_topic_document = [os.path.join(file_path_topic_document, f) 
                      for f in os.listdir(file_path_topic_document) 
                      if f.endswith(".csv")]


# In[ ]:

#gpuN="0"#sys.argv[1]
#os.environ['CUDA_VISIBLE_DEVICES'] = gpuN
#gpu_name='/gpu:'+gpuN


# In[ ]:

import subprocess, re, os, sys #https://github.com/yaroslavvb/stuff/blob/master/notebook_util.py
def run_command(cmd):
    """Run command, return output as string."""
    
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


# In[ ]:

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


# In[ ]:

def save_txt(txt,file):
    text_file = open(file, "w")
    text_file.write(txt)
    text_file.close()


# In[ ]:

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




# In[ ]:




# In[36]:

def read_cvs_by_pands(path_database,file_database,index_col, header):
    import csv
    import pandas as pd
    return pd.read_csv(path_database+file_database,index_col=index_col,header=header)



# In[37]:

def read_cvs_by_pands_full_path(full_path,index_col, header):
    import csv
    import pandas as pd
    return pd.read_csv(full_path,index_col=index_col,header=header)
 


# In[38]:

def save_df_to_csv(df,path_database,sub_path,new_file_name):
     df.to_csv(path_database+sub_path+new_file_name)


# In[39]:

def df_to_dict(df):
        
    dic={}
    keys=df.keys()
    #print(keys)
    
    values= df.T.values.tolist()
    #print(len(values))
    for i in range(len(keys)):
        #print(keys[i])
        dic[keys[i]]=magic(values[i])
    return dic


# In[40]:

def magic(numList):         # [1,2,3]
    s = map(str, numList)   # ['1','2','3']
    s = ''.join(s)          # '123'
    s = int(s)              # 123
    return s


# In[41]:

def list_to_dict(list_full_path):
    dic={}
    for file_path in list_full_path:
        index=full_name_file(file_path)
        dic[index]=file_path

    return dic
    


# In[42]:

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


# In[ ]:

def isfile_empty(file_path_name):
    f=open(file_path_name, 'r') 
    is_blank = len(f.read().strip()) == 0
    return is_blank


# In[43]:

def full_name_file(full_name_path):#like cs.csv
    d=full_name_path.split("/")
    ##print(d)
    name=d[len(d)-1]#.split(".")
    return name


# In[44]:

def read_last_file_list(file_path,extention):
    
    file_names = [os.path.join(file_path, f) 
                      for f in os.listdir(file_path) 
                      if f.endswith(extention) and not isfile_empty(file_path+f)]
    return file_names


# In[45]:

#permutation
def permutation(path_data_base,path_sim,path_sim_permutation):
    # open file Sim 
    sim_list=read_last_file_list(path_data_base+path_sim,"csv")
    #print(len(sim_list))
    for y in range(len(sim_list)):

        file_name=full_name_file(str(sim_list[y]))
        #print(file_name)
        Topic_name=file_name[:-4]
        #print(Topic_name)
        topic_df=read_cvs_by_pands(path_data_base,path_sim+file_name,None
                          ,None)


        #print(len(topic_df))
        main_Topic=[]
        sub_topic=[]
        true_list=[]
        list_removed=[]
        for i in range (len(topic_df)):
            #print(i)
            #print(topic_df.iloc[i][0])
            if topic_df.iloc[i][0]==Topic_name:
                #exclud from list wordwith the same name file
                main_Topic=list(topic_df.iloc[i])
                true_list.append(main_Topic)
                #print(type(main_Topic))
                #print(main_Topic)
                sub_topic =topic_df[topic_df.index != i].values.tolist()
                break

                # list after remove main topic
        if len(sub_topic)>1:
            for term in range(len(sub_topic)):
                if term not in list_removed:

                    #print("-------------------------------------")
                    #print("============Term==========",sub_topic[term][0])

                    is_term_good=True
                    for next_term in sub_topic[term+1:]:

                        #print(next_term[0])
                        result=similarity_by_infocontent(sub_topic[term][0],next_term[0],'res')
                        if result<1:
                            if sub_topic[term][1]>next_term[1]:
                                list_removed.append(next_term[0])
                            else:
                                list_removed.append(sub_topic[term][0])

                            is_term_good=False
                            #print(sub_topic[term][0],next_term[0],result,"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                            break
                    if is_term_good:
                        true_list.append(sub_topic[term])
                else:
                    print("------------term removed-------------------------",term)
            #print(true_list)
            perm_df=pd.DataFrame(true_list)
            #print(" print new file to perm")
            save_df_to_csv(perm_df,path_data_base,path_sim_permutation,file_name)
        else:
            #print(" copy origenal file to perm")
            save_df_to_csv(topic_df,path_data_base,path_sim_permutation,file_name)


        #compare each item withothers 
        #each term with next term 
        #if relation is less than one 
        #remove term with origenal small value
        #and new list in next compare 


# In[46]:

def log_idf(D,d,base):
    return math.log((D/d), base)


# In[47]:

def idf_df(df,D,base):
    #[7/df['0']]
    y = [log_idf(D,x,base) for x in df[0]]
    return y


# In[48]:

def read_sord_tf_file(path_data_source,sub_path_data_source_tf,i):
    #df=read_cvs_by_pands(path_data_source,sub_path_data_source_tf+"cs"+str(i)+".csv",0,header=0)
    #print("nnnnnnnnnnnn",path_data_source+sub_path_data_source_tf+i)
    df=read_cvs_by_pands(path_data_source,sub_path_data_source_tf+i,0,header=0)
    #ee.T.sort_index(inplace=True)
    df = df.T.sort_index()
    return df


# In[49]:

def read_sord_idf_file(path_data_source,sub_path_data_source_idf,i):
    #df=read_cvs_by_pands(path_data_source,sub_path_data_source_idf+"cs"+str(i)+".csv",0,header=0)
    df=read_cvs_by_pands(path_data_source,sub_path_data_source_idf+i,0,header=0)
    #ee.T.sort_index(inplace=True)
    df = df.T.sort_index()
    return df


# In[50]:

#d1={'a': 1, 'b': 1, 'c': 1}
def dict_to_DF(dic):
    df=pd.DataFrame([dic])
    return df

#dict_to_DF(d1)


# In[51]:

def save_df_to_csv(df,path_database,sub_path,new_file_name):
     df.to_csv(path_database+sub_path+new_file_name)


# In[52]:

def tf_idf_one(path_data_source,sub_path_data_source_tf,sub_path_data_source_idf,i,sub_path_data_source_tfidf):
    df_tf=read_sord_tf_file(path_data_source,sub_path_data_source_tf,i)
    df_idf=read_sord_idf_file(path_data_source,sub_path_data_source_idf,i)

    if len(df_idf) != 0:
        idf=idf_df(df_idf,len(df_idf),10)

        full_tfidf=[]
        for index in range(len(idf)):
            #print(index)
            tfidf=df_tf[0][index]*idf[index]
            full_tfidf.append(tfidf)


    df_tf_idf=pd.DataFrame(full_tfidf)
    df_tf_idf.index=df_tf.index
    #df_tf_idf.to_csv(path_data_source+sub_path_data_source_tfidf+"cs"+str(i)+".csv")
    df_tf_idf.to_csv(path_data_source+sub_path_data_source_tfidf+i)


# In[53]:

def tf_idfs(list_tf,list_idf):
    #print(list_tf,list_idf)
    for tf_file in list_tf:
        tf_file_name=full_name_file(tf_file)
        #print(tf_file_name)
        for idf_file in list_idf:
            idf_file_name=full_name_file(idf_file)
            if tf_file_name == idf_file_name:
                #print(idf_file_name)
                tf_idf_one(path_data_base,path_tf,path_idf,tf_file_name,path_tfidf)
            
        
        


# In[72]:

#to calculate H
#load list V(TF-IDF)
#loop perfile
#hold term
#open  the same name in topic "permutation"
#scalar multiblication: multible each 
#value of this word in V file by all items (terms)
#in topic file then sum and put result  in new file for topic document 
def W_topic_doc(file_names_tfidf,file_path_sim_permutation,path_topic_document,path_data_base):
    premu_dic=list_to_dict(file_path_sim_permutation)
    tfidf_dict=list_to_dict(file_names_tfidf)
    #print(tfidf_dict)
    
    for tfidf_file in file_names_tfidf:
        
        file_name=full_name_file(tfidf_file)
        #print("File: ",file_name,"\n")
        tfidf_df=read_cvs_by_pands_full_path(tfidf_file,0,0)
        df_index=tfidf_df.index
        result_W={}
        for index in df_index:
            #print("Term will be Topic: ",index,"\n")
            full_topic_path=premu_dic.get(index+'.csv')
            #print(full_topic_path)
            topic_df=read_cvs_by_pands_full_path(full_topic_path,0,0)
            #print("fffffffffffffff",tfidf_df.loc[index][0])
            
            #print("Topic-Terms:\n",topic_df['0'],"\n")
            sum_terms_topic=sum([tfidf_df.loc[index][0] * x for x in topic_df['1']])
            result_W[index]=sum_terms_topic
            #print("Sum: ",sum_terms_topic,"\n")
        W_df=dict_to_DF(result_W)
        #print("Save",file_name)
        save_df_to_csv(W_df,path_data_base,path_topic_document,file_name)
        #print("All: ",result_W,"\n")
#return result_W


# In[1]:

def gpu_full_process(file_name):
    
    #if gpu_name=='/gpu:0':
        
        #with tf.device(gpu_name):
            #print("in gpu:0")

    #file_names_tf=read_last_file_list(file_path_tf,"csv")
    #file_names_idf=read_last_file_list(file_path_idf,"csv")        
    #tf_idfs(file_names_tf,file_names_idf)
    tf_idf_one(path_data_base,path_tf,path_idf,file_name,path_tfidf)
    
    #file_names_tfidf=read_last_file_list(file_path_tfidf,"csv")
    
            
    '''if gpu_name=='/gpu:1':    
        with tf.device(gpu_name):
            print("in gpu:1")

            permutation(path_data_base,path_sim,path_sim_permutation)
            file_names_sim_permutation=read_last_file_list(file_path_sim_permutation,"csv")'''

    #with tf.device('/gpu'):
        
        #W_topic_doc(file_names_tfidf,file_names_sim_permutation,path_topic_document,path_data_base)
    
    
    
    
    


# In[2]:

#gpu_full_process()


# In[ ]:

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
    


# In[30]:

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# In[31]:

config = tf.ConfigProto(device_count={processor.upper():int(all_gpus)},
                        allow_soft_placement=True,
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1,
                        use_per_session_threads=True,
                        log_device_placement=True)
config.gpu_options.allow_growth = True


# In[ ]:

def distrupited_task_gpu(no_task,total_no_gpu):
    
    return no_task%total_no_gpu


# In[32]:


#start_gpu_memory=gpu_memory_map("gpu_befor_session_memory.txt","gpu_befor_session_out.txt")
with tf.Session(config=config) as sess:
    print("************Started Session CPU *************")
    sess.run(tf.global_variables_initializer())
    #start_gpu_memory=gpu_memory_map("gpu_after_start_session_memory.txt","gpu_after_start_session_out.txt")
    #print("gpu_memory_map_after_session",start_gpu_memory) 
    gpu_name='/'+processor+':0'
    with tf.device(gpu_name):
        #with tf.device('/cpu'):
        
        #gpu_no=distrupited_task_gpu(int(task_no),int(all_gpus))
        #gpu_name='/gpu:'+str(gpu_no)
        
        #with tf.device(gpu_name):
        filename=dataset+task_no+".csv" #"cs"+task_no+".txt" 
        gpu_full_process(filename)
        
           

        #with tf.device(gpu_name):
            #start = timeit.default_timer()
        
            #Your statements here
            #print(addd("1","2"))
            #stop = timeit.default_timer()
            #print('Total Time:',stop)


# In[ ]:



