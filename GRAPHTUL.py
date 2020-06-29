# -*- coding:  UTF-8 -*-
from __future__ import division
import math
from tensorflow.python.layers.core import Dense
import seq2seq_c as seqc
#from metric import *
from ops import *
import time
import datetime
import collections
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from metrics import *
# =============================== vars ====================================== #
initializer=tf.truncated_normal_initializer(stddev=0.02)
n_hidden=300
batch_size=16
user_size=300
embedding_size=250
epoch=50
# =============================== data load ====================================== #
#load original data
dataname=['gowalla','brightkite','NYC','TKY']
data_choose=3
training_rate=0.5
label_size=300 #user number
file_load=open('./data/'+dataname[data_choose]+'_scopus.dat','r')
print 'start load data'+dataname[data_choose]
O_Data=collections.OrderedDict()
tra_num=0
for lines in file_load.readlines():
    lineArr = lines.split()
    tra_num+=1
    O_Data.setdefault(int(lineArr[0]),[]).append(lineArr)
print 'finish load data'+dataname[data_choose]
print 'Total user number of this dataset is '+str(len(O_Data))
print 'Total trajectory number of this dataset is '+str(tra_num)
initial_label_num=0
Training_SET=[]
Testing_SET=[]
for keys in O_Data.keys():

    if initial_label_num>=label_size:
        break
    User_Data=O_Data[keys]
    if len(User_Data)<2: #this trajectory only one a bug
        Training_SET.append(User_Data[0])
        initial_label_num += 1
        continue
    split_number=int(math.ceil(len(User_Data)*training_rate))
    #print len(User_Data),split_number
    for value_train in User_Data[:split_number]:
        Training_SET.append(value_train)
    for value_test in User_Data[split_number:]:
        Testing_SET.append(value_test)
    #print keys
    initial_label_num+=1
#print Training_SET
print 'training size, testing size',len(Training_SET),len(Testing_SET)
poi_voc=list()
poi_count={}
for tuple in (Training_SET+Testing_SET):
    if len(tuple)<2:
        print 'bug!'
    for poi in tuple[1:]: #remove user id
        poi_voc.append(poi)
        poi_count.setdefault(poi,[]).append(poi)

dictionary={}
for key in poi_count.keys():
    count=len(poi_count[key])
    dictionary[key]=count
dictionary['<GO>']=1
dictionary['<PAD>']=1
dictionary['<EOS>']=1
new_dict=sorted(dictionary.items(),key = lambda x:x[1],reverse = True)
voc_poi=list()
for item in new_dict:
    voc_poi.append(item[0]) #has been sorted by frequency


poi_voc=list(set(poi_voc))
print 'POI number is',len(poi_voc)
poi_voc.append('<EOS>')
poi_voc.append('<GO>')
poi_voc.append('<PAD>')
#embeddings
table_X={} #trajectory
new_table_X={}
def getXs():  # 读取轨迹向量
    fpointvec = open('./pre_data/TKY_embeddings_graph.dat', 'r')  # 获取check-in向量 已经用word2vec gowalla_em_250 TKY_embeddings_graph.dat
    #     table_X={}  #建立字典索引gowalla_user_vector250d_   gowalla_user_vector250d_.dat   GW_embeddings
    item = 0
    for line in fpointvec.readlines():
        lineArr = line.split()
        if (len(lineArr) < 250): #delete fist row
            continue
        item += 1  # 统计条目数
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  # 读取向量数据
        if lineArr[0] == '</s>': #</s>  9999
            table_X['<PAD>']=X  #dictionary is a string  it is not a int type
        else:
            table_X[lineArr[0]] =X
    print "point number item=", item
    return table_X
table_X=getXs()
for poi in poi_voc:
    new_table_X[poi]=table_X[poi]
new_table_X['<GO>']=table_X['<GO>']
new_table_X['<EOS>']=table_X['<EOS>']
new_table_X['<PAD>']=table_X['<PAD>']
#Get dictionary
def dic_em():
    dic_embeddings=list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
    return dic_embeddings
dic_embeddings=tf.constant(dic_em())
weights = tf.Variable(tf.truncated_normal([embedding_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
bias = tf.Variable(tf.zeros([embedding_size]), dtype=tf.float32)
dic_embeddings=tf.nn.relu(tf.nn.xw_plus_b(dic_embeddings, weights, bias)) #


voc_tra=list()
for keys in new_table_X:
    voc_tra.append(keys)


def extract_words_vocab():
    int_to_vocab = {idx: word for idx, word in enumerate(voc_tra)} #voc_poi
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
int_to_vocab, vocab_to_int=extract_words_vocab()

def get_label_value(vocab_label_set):
    Label = list(set(vocab_label_set)) #get label set
    User_List = sorted(Label)
    number_list = list(map(int, list(Label)))
    #User_List=sorted(number_list)
    return User_List
New_Training_SET=list()
New_Testing_SET=list()
Train_Label=list()
Test_Label=list()

for item in Training_SET:
    value=item[1:]
    Train_Label.append(int(item[0])) #add train user label char
    temp = list()
    for poi in value:
        temp.append(vocab_to_int[poi])
    New_Training_SET.append(temp)

for item in Testing_SET:
    value=item[1:]
    Test_Label.append(int(item[0])) #add test user label char
    temp = list()
    for poi in value:
        temp.append(vocab_to_int[poi])
    New_Testing_SET.append(temp)


Label=get_label_value(Train_Label+Test_Label)
for i in range(len(Label)-1):
    if Label[i]>Label[i+1]:
        print 'user error!!'
print 'Trainning User number is ',len(Label)
def get_mask_index(value, User_List): #get mask id #int
    return User_List.index(value)
def get_true_index(index, User_List): #get real id #int
    return User_List[index]


#------------------------START Obtain dataset including training and testing----------------
#sort trajectory
# sort train
index_Train = {}
new_trainT = []
new_trainU = []
for i in range(len(New_Training_SET)):
    index_Train[i] = len(New_Training_SET[i])
temp_size = sorted(index_Train.items(), key=lambda item: item[1])
for i in range(len(temp_size)):
    id = temp_size[i][0]
    new_trainT.append(New_Training_SET[id])
    new_trainU.append(get_mask_index(Train_Label[id], Label)) #mask id


#sort trajectory
# sort test
index_Test = {}
new_testT = []
new_testU = []
for i in range(len(New_Testing_SET)):
    index_Test[i] = len(New_Testing_SET[i])
temp_size = sorted(index_Test.items(), key=lambda item: item[1])
for i in range(len(temp_size)):
    id = temp_size[i][0]
    new_testT.append(New_Testing_SET[id])
    new_testU.append(get_mask_index(Test_Label[id],Label))
#------------------------END Obtain dataset including training and testing

#POI embedding
initial_embeddings = tf.Variable(tf.random_uniform([len(voc_poi), embedding_size], -1.0, 1.0))
weights = tf.Variable(tf.truncated_normal([embedding_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
bias = tf.Variable(tf.zeros([embedding_size]), dtype=tf.float32)
embeddings = tf.nn.xw_plus_b(initial_embeddings, weights, bias)



# =============================== data load end====================================== #

#define functions
def get_onehot(index): # one-hot
    x = [0] * label_size
    x[index] = 1
    return x
# =============================== tf.vars ====================================== #
keep_prob=tf.placeholder("float")
it_learning_rate=tf.placeholder("float")
encoder_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
decoder_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
label_y=tf.placeholder(dtype=tf.float32,shape=[batch_size,label_size])
target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
#define the weight and bias dictionary
with tf.name_scope("weight_inital"):
    weights={
        'predictor_w':tf.Variable(tf.random_normal([n_hidden,user_size],mean=0.0, stddev=0.01)),
    }
    biases= {
    'predictor_b': tf.Variable(tf.random_normal([user_size], mean=0.0, stddev=0.01)),
    }
# =============================== Encoder ====================================== #
def encoder(X,keep_prob=0.5):
    """
    encode discrete feature to continuous latent vector
    :param tensor: [batch_size,length,embedding_size].
    :return:encoded latent vector
    """
    with tf.variable_scope("encoder"): #dic_
        tensor=tf.nn.embedding_lookup(dic_embeddings,X) #find embeddings of trajectory:[batch_size,length,embedding_size].
        trans_tensor=tf.transpose(tensor,[1,0,2])       #[length,batch_size,embedding_size].
        lstm_cell=tf.nn.rnn_cell.LSTMCell(n_hidden)
        dr_lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=keep_prob)
        (output,states)=tf.nn.dynamic_rnn(dr_lstm_cell,trans_tensor,time_major=True,dtype=tf.float32)
        latent_code=output[-1]
        #latent_code=tf.reduce_mean(output)
        latent_code=tf.nn.l2_normalize(latent_code)
        print 'latentcode',latent_code
        #latent_code= fully_connected(latent_code, c_dim, initializer=initializer, is_last=True, scope="encoder_output")
        return latent_code,states

# =============================== Decoder ====================================== #
def decoder(tensor,X,en_state,reuse=False):
    """
     decode continuous vector
     """
    with tf.variable_scope('decoder',reuse=reuse) as scope:

        decode_lstm=tf.nn.rnn_cell.LSTMCell(n_hidden)
        decode_dr_lstm = tf.nn.rnn_cell.DropoutWrapper(decode_lstm, output_keep_prob=keep_prob)
        output_layer=Dense(len(vocab_to_int))
        decoder_initial_state=en_state #LSTMStateTuple(c_state, h_state)

        copy = tf.tile(tf.constant([vocab_to_int['GO']]), [batch_size])
        training_helper = seqc.GreedyEmbeddingHelper2(embeddings,
                                                         sequence_length=target_sequence_length, start_tokens=copy)
        training_decoder = seqc.BasicDecoder(decode_dr_lstm, training_helper, decoder_initial_state,tensor,output_layer)  # cell,helper, initial_state, out_layer(convert rnn_size to vocab_size)
        output, _, _ = seqc.dynamic_decode(training_decoder,
                                              impute_finished=True,
                                              maximum_iterations=max_target_sequence_length)

        predicting_logits = tf.identity(output.sample_id, name='predictions')
        training_logits = tf.identity(output.rnn_output, 'logits')
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        target = X
        return output, predicting_logits, training_logits, masks, target

# =============================== Generator ====================================== #
def generator(z,context,reuse=False):
    """
    generator of WGAN
    :param z: 2-D tensor. noise with standard normal distribution
    :param reuse: Boolean. reuse or not
    :return: 2-D tensor. latent vector
    """
    """
    encode discrete feature to continuous latent vector
    :param tensor: [batch_size,length,embedding_size].
    :return:encoded latent vector
    """
    with tf.variable_scope("generator"):
        flatent_code=z
        flatent_code=tf.nn.l2_normalize(flatent_code)
        return flatent_code


# =============================== Discriminator(Critic) ====================================== #
def critic(latent,reuse=False):
    """
    discriminator of WGAN
    :param latent: 2-D tensor. latent vector
    :param reuse: Boolean. reuse or not
    :return: 2-D tensor. logit of data or noise
    """
    with tf.variable_scope("critic",reuse=reuse):
        fc_100 = fully_connected(latent, 100, initializer=initializer, scope="fc_100")
        fc_60 = fully_connected(fc_100, 60, initializer=initializer, scope="fc_60")
        fc_20 = fully_connected(fc_60, 20, initializer=initializer, scope="fc_20")
        output=fully_connected(fc_20,1,initializer=initializer,is_last=True,scope="critic_output")
        #WGAN does not using activate
    return  output

# =============================== Predictor ====================================== #

def predictor(latent_code,reuse=False):
    with tf.variable_scope("predictor", reuse=reuse):
        pred = (tf.matmul(latent_code, weights["predictor_w"]) + biases["predictor_b"])
        return pred
# =============================== Function ====================================== #
def autoencoder(X,de_X,context,keep_prob):
    """
    deep autoencoder. reconstruction the input data
    :param data: 2-D tensor. data for reconstruction
    :return: 2-D tensor. reconstructed data and latent vector
    """
    with tf.variable_scope("autoencoder"):
        latent,en_state=encoder(X,context,keep_prob)
        output_, predicting_logits_, training_logits_, masks_, target_=decoder(latent,de_X,en_state)
    return training_logits_,masks_,target_,latent,predicting_logits_,en_state
def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])  # 取最大长度
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def pad_time_batch(time_batch):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in time_batch])  # 取最大长度
    return [sentence + [0] * (max_sentence - len(sentence)) for sentence in time_batch]
def pad_dist_batch(dist_batch):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in dist_batch])  # 取最大长度
    return [sentence + [sentence[-1]] * (max_sentence - len(sentence)) for sentence in dist_batch]
def eos_sentence_batch(sentence_batch, eos_in):
    return [sentence + [eos_in] for sentence in sentence_batch]

#=============================== Graph ====================================== #
"""
build network
:return:
"""
#TUL
latent_space,states=encoder(X=encoder_input,keep_prob=keep_prob)
pred=predictor(latent_space) #no softmax
soft_pred=tf.nn.softmax(pred) #add softmax
#classifier loss
classifer_cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=label_y))

#Opt
optimizer=tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(classifer_cost)
#new opt
encoder_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="encoder")
predictor_variables=[weights["predictor_w"],weights,bias,biases["predictor_b"]]
updated_classifier=tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(classifer_cost)
#evaluate model

correct_pred=tf.equal(tf.arg_max(pred,1),tf.arg_max(label_y,1)) #1表示维度
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#=============================== Train ====================================== #
"""
train network
:return:
"""

def train():
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    debug=True
    with tf.Session() as sess:
        #sess.run(init)
        saver.restore(sess, './temp/tky_initial_graphtul.pkt')
        initial_learning_rate = 0.00025
        for iter in range(epoch): #iteration time
            step = 0 #batch train
            #Record Data
            Train_Predicted_Value=[]
            Train_User_Value=[]
            while step < len(new_trainT) // batch_size: # it does not include the last part the size is smaller than batch_size
                start_i = step * batch_size
                label_input_x = new_trainT[start_i:start_i + batch_size]
                # 补全序列
                encoder_input_batch = pad_sentence_batch(label_input_x, vocab_to_int['<PAD>']) # add pad
                decoder_input_batch=eos_sentence_batch(label_input_x,vocab_to_int['<EOS>'])
                decoder_input_batch = pad_sentence_batch(decoder_input_batch, vocab_to_int['<PAD>'])
                label_Y_mask_id = []
                # label data length
                laebl_source_lengths = []
                for source in label_input_x :
                    laebl_source_lengths.append(len(source)+1) # add a end flag size=size+1
                #get batch_user
                for l_y_i in range(start_i,start_i + batch_size):
                    Train_User_Value.append(new_trainU[l_y_i]) #mask_id
                    label_y_i=get_onehot(new_trainU[l_y_i])#
                    label_Y_mask_id.append(label_y_i)

                #TUL run
                feed_dict={encoder_input:encoder_input_batch,label_y:label_Y_mask_id,it_learning_rate:initial_learning_rate,keep_prob:0.5}
                opt,out_pred=sess.run([updated_classifier,soft_pred],feed_dict=feed_dict) #pred is no softmax updated_classifier
                #print out_pred
                for each_out_pred in out_pred:
                    Train_Predicted_Value.append(each_out_pred)
                step+=1 # add step


            #Calculation
            train_acc1,train_acc5=accuracy_K(pred=Train_Predicted_Value,label=Train_User_Value)
            macro_f1,macro_r,macro_p=macro_F(pred=Train_Predicted_Value,label=Train_User_Value)
            print 'training->iteration=',iter,'acc@1=',train_acc1,'acc@5',train_acc5,'macro-f1',macro_f1

            #TEST
            #print 'testing item',len(new_testT)
            saver.save(sess, './temp/tky_initial_graphtul.pkt')
            test(sess,new_testT,new_testU)



def test(sess,new_testT,new_testU):
    # Record Data
    Test_Predicted_Value = []
    Test_User_Value = []
    step=0
    while step < len(new_testT) // batch_size:  # it does not include the last part the size is smaller than batch_size
        start_i = step * batch_size
        label_input_x = new_testT[start_i:start_i + batch_size]
        # 补全序列
        encoder_input_batch = pad_sentence_batch(label_input_x, vocab_to_int['<PAD>'])  # add pad
        decoder_input_batch = eos_sentence_batch(label_input_x, vocab_to_int['<EOS>'])
        decoder_input_batch = pad_sentence_batch(decoder_input_batch, vocab_to_int['<PAD>'])
        label_Y_mask_id = []
        # label data length
        laebl_source_lengths = []
        for source in label_input_x:
            laebl_source_lengths.append(len(source) + 1)  # add a end flag size=size+1
        # get batch_user
        for l_y_i in range(start_i, start_i + batch_size):
            Test_User_Value.append(new_testU[l_y_i])  # mask_id

            label_y_i = get_onehot(new_testU[l_y_i])  #
            label_Y_mask_id.append(label_y_i)

        # TUL run
        feed_dict = {encoder_input: encoder_input_batch,
                      keep_prob: 1.0}
        out_pred = sess.run(soft_pred, feed_dict=feed_dict)  # pred is no softmax
        # print out_pred
        for each_out_pred in out_pred:
            Test_Predicted_Value.append(each_out_pred)
        step += 1  # add step

    # Calculation
    test_acc1, test_acc5 = accuracy_K(pred=Test_Predicted_Value, label=Test_User_Value)
    macro_f1, macro_r, macro_p = macro_F(pred=Test_Predicted_Value, label=Test_User_Value)
    print 'testing->'+'acc@1=', test_acc1, 'acc@5', test_acc5,'macro-p',macro_p,'macro-r',macro_r,'macro-f1',macro_f1,
if __name__ == "__main__":
    train()
    print 'Model END'