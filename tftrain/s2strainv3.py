# -*- coding: utf-8 -*-

import sys
import math
import tflearn
import tensorflow as tf
#from tensorflow.python.ops import rnn_cell
#from tensorflow.python.ops import rnn
#import chardet
import numpy as np
import struct

#------------------------------------------------------------------------------#


max_w = 50
float_size = 4
word_vec_dim = 200
max_seq_len = 8

## 切分词汇表中的单词字典
vocab_dictionary = {}
## 词向量中的向量字典
vector_dictionary = {}

## 向量字典中question单词的KV键值的值列表
question_list = []
## 向量字典中answer单词的KV键值的值列表
answer_list = []

## epoch太大，耗时太长
max_epoch_num = 2

## 几个预定义的文件
vocab_file = 'xhj.seg'
test_seg = 'xhjtest.seg'
vector_file = 'xhjvec.bin'


#------------------------------------------------------------------------------#

'''
load_vocabulary
加载原始切词语料中的文本词汇
这个方法填充vocab_dictionary字典
    key = 单词
  value = 1 ## 没有实际意义的值
'''


def load_vocabulary():
    print 'load vocabulary begin'
    f = open(vocab_file, 'r')
    while True:
        line = f.readline()
        if line:
            line_pair = line.split('|')
            line_question = line_pair[0]
            line_answer = line_pair[1]
            for word in line_question.decode('utf-8').split(' '):
                vocab_dictionary[word] = 1  ## KV： k = word, v = 1
            for word in line_answer.decode('utf-8').split(' '):
                vocab_dictionary[word] = 1  ## KV： k = word, v = 1
        else:
            break
    f.close()
    print 'load vocabulary end'


'''
load_bin_vectors
一次性加载词向量文件中的词向量。
这里的词向量文件是word2vec生成的binary文件。

这个文件的第一行是“6827 200”,
  6827：表示词向量数量。每行一个词向量
   200：表示每个向量的维度大小。这个大小是word2vec参数size指定的。


这个方法填充词向量字典vector_dictionary
    key = 单词
  value = 200维向量

词向量格式：
你 0.154262 -0.947298 -0.671668 -0.560019 -0.084487 -0.126347 0.251861 0.440904 -0.269444 -0.112964 0.108902 0.373456 0.072086 -0.194668 -0.031037 -0.082331 0.122092 0.033872 0.416037 0.055250 0.455487 -0.005934 0.085576 -0.565697 -0.072866 -0.268784 -0.383227 0.107736 0.167225 -0.179840 -0.219223 -0.223419 0.000891 0.621202 0.013182 -0.224104 -0.165722 0.242460 0.105830 0.270964 -0.297463 -0.178786 0.342809 -0.489887 0.584078 -0.017644 -0.068940 -0.518581 -0.436711 0.201550 0.316783 0.154611 0.189381 0.484812 -0.089577 0.275189 0.080310 0.047296 0.336020 -0.075151 -0.326348 -0.133550 -0.164151 -0.365974 -0.002789 -0.420074 0.299372 -0.073010 0.186314 0.255517 0.101990 0.197863 0.242400 -0.043798 -0.131116 0.430205 -0.240594 0.064068 0.102266 -0.324403 0.073527 -0.112213 -0.061090 -0.031712 0.371045 0.470470 0.475879 0.207059 0.370857 0.007299 0.064741 -0.274955 -0.070386 0.201195 0.199712 -0.057381 0.311169 0.314671 -0.390109 -0.140684 0.550947 -0.207940 -0.119161 0.470585 0.179260 0.247981 -0.389022 -0.094208 -0.106551 0.117773 -0.089252 0.239677 -0.102344 -0.253826 -0.172784 0.264334 0.072033 -0.076909 -0.302443 -0.110471 0.138127 -0.039360 0.537022 0.185650 -0.016361 0.172388 0.046214 0.356799 0.060380 -0.158818 -0.396843 0.302571 0.057653 -0.231728 0.092829 0.506258 0.076772 -0.223876 -0.029504 -0.051348 0.002130 0.072459 0.153248 0.153057 0.131514 0.296695 -0.140707 0.218606 -0.551975 -0.409876 0.015616 -0.035830 0.407998 0.065962 0.154129 -0.357424 0.386741 -0.311028 -0.207643 0.194520 -0.073731 -0.046240 0.010608 0.072246 -0.578408 0.366124 -0.395570 -0.057476 0.245619 -0.050978 0.225124 -0.329954 -0.461468 -0.161946 -0.073585 -0.183873 -0.150828 -0.501302 0.089463 -0.122170 0.133381 -0.136169 0.555761 0.516429 -0.222246 0.366125 0.186855 0.401108 0.234727 0.232667 0.270441 -0.024804 -0.051669 -0.243818 0.075833 0.056143 -0.661869 0.177979 -0.101192 -0.031108 

'''


def load_bin_vectors():
    print "begin load vectors " + vector_file
    f = open(vector_file, "rb")

    # 获取词表数目及向量维度
    first_line = f.readline()
    first_line = first_line.strip()
    words_num = long(first_line.split(' ')[0])
    dim_size = long(first_line.split(' ')[1])
    print "words number =", words_num
    print "dimension size =", dim_size

    for b in range(0, words_num):  # 逐个取词向量
        a = 0
        word = ''
        print '\n\n>>>>>>>>>>>>>>>>>>>>>'
        while True:
            c = f.read(1)  # 每行第一个是词，逐个读取单字，拼装。
            word = word + c
            print ':: ' + word
            if False == c or c == ' ':
                break
            if a < max_w and c != '\n':  # 单词最大不超过max_w个字
                a = a + 1
        word = word.strip()  # 整个词存在这里
        print 'word: ' + word

        vector = []
        for index in range(0, dim_size):
            m = f.read(float_size)
            (weight,) = struct.unpack('f', m)
            vector.append(float(weight))

        # 将词及其对应的向量存到dict中
        if vocab_dictionary.has_key(word.decode('utf-8')):
            vector_dictionary[word.decode('utf-8')] = vector[0:word_vec_dim]

    f.close()
    print "load vectors finish"


'''
init_seq
初始化
从文件中切分出question序列和answer序列

:input_file: 
  经过切词后的训练文件或者测试文件
  语料格式必须是空格分词的，例如：“你 是 谁 | 我 是 小 萝卜”
:return: None
'''


def init_seq(seg_file):
    print 'init Q/A sequences from: ' + seg_file
    f = open(seg_file, 'r')
    while True:
        question_seq = []
        answer_seq = []
        line = f.readline()
        if line:
            qalist = line.split('|')
            question = qalist[0]
            answer = qalist[1]
            for word in question.decode('utf-8').split(' '):
                if vector_dictionary.has_key(word):
                    question_seq.append(vector_dictionary[word])
            for word in answer.decode('utf-8').split(' '):
                if vector_dictionary.has_key(word):
                    answer_seq.append(vector_dictionary[word])
        else:
            break
        question_list.append(question_seq)
        answer_list.append(answer_seq)

    f.close()
    print 'load all sequences over'


'''
vector_sqrtlen
:vector: 
:return: lenth
'''


def vector_sqrtlen(vector):
    len = 0
    for item in vector:
        len += item * item
    len = math.sqrt(len)
    return len


'''
vector_cosine
:v1, v2: 
:return: cosine value
'''


def vector_cosine(v1, v2):
    if len(v1) != len(v2):
        sys.exit(1)
    sqrtlen1 = vector_sqrtlen(v1)
    sqrtlen2 = vector_sqrtlen(v2)
    value = 0
    for item1, item2 in zip(v1, v2):
        value += item1 * item2
    return value / (sqrtlen1*sqrtlen2)


'''
vector2word
:param, vector: 
:return: words
'''


def vector2word(vector):
    max_cos = -10000
    match_word = ''
    for word in vector_dictionary:
        v = vector_dictionary[word]
        cosine = vector_cosine(vector, v)
        if cosine > max_cos:
            max_cos = cosine
            match_word = word
    return (match_word, max_cos)

#------------------------------------------------------------------------------#


'''
Class MySeq2Seq

    seq2seq模型
    思路：
    输入输出序列一起作为input，然后通过slick和unstack切分。
    完全按照论文说的编码器解码器来做，输出的时候把解码器的输出按照词向量的200维展平，这样输出就是(?,seqlen*200)
    这样就可以通过regression来做回归计算了，输入的y也展平，保持一致
'''


class MySeq2Seq(object):

    '''
    init constructor
    :param input_file: 切词后的词汇文件
    '''


    def __init__(self, max_seq_len = 16, word_vec_dim = 200, input_file = vocab_file):
        self.max_seq_len = max_seq_len
        self.word_vec_dim = word_vec_dim
        self.input_file = input_file


    '''
    generate_training_data
        加载词汇表
        加载词向量表
        初始化Quesetion/Answer序列
    :returns : 
    '''


    def generate_training_data(self):
        load_vocabulary()
        load_bin_vectors()
        init_seq(self.input_file)
        xy_data = []
        y_data = []
        for i in range(len(question_list)):
            question = question_list[i]
            answer = answer_list[i]
            if len(question) < self.max_seq_len and len(answer) < self.max_seq_len:
                sequence_xy = [np.zeros(self.word_vec_dim)] * (self.max_seq_len-len(question)) + list(reversed(question))
                sequence_y = answer + [np.zeros(self.word_vec_dim)] * (self.max_seq_len-len(answer))
                sequence_xy = sequence_xy + sequence_y
                sequence_y = [np.ones(self.word_vec_dim)] + sequence_y
                xy_data.append(sequence_xy)
                y_data.append(sequence_y)

                #print "right answer"
                #for w in answer_seq:
                #    (match_word, max_cos) = vector2word(w)
                #    if len(match_word)>0:
                #        print match_word, vector_sqrtlen(w)

        return np.array(xy_data), np.array(y_data)



    '''
    model
      建模，训练模型或者测试模型
      
      ：param  feed_previous: 喂数据
      :returns : model
    '''


    def model(self, feed_previous=False):
        # 通过输入的XY生成encoder_inputs和带GO头的decoder_inputs
        input_data = tflearn.input_data(shape=[None, self.max_seq_len*2, self.word_vec_dim], dtype=tf.float32, name = "XY")
        encoder_inputs = tf.slice(input_data, [0, 0, 0], [-1, self.max_seq_len, self.word_vec_dim], name="enc_in")
        decoder_inputs_tmp = tf.slice(input_data, [0, self.max_seq_len, 0], [-1, self.max_seq_len-1, self.word_vec_dim], name="dec_in_tmp")
        go_inputs = tf.ones_like(decoder_inputs_tmp)
        go_inputs = tf.slice(go_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
        decoder_inputs = tf.concat([go_inputs, decoder_inputs_tmp], 1, name="dec_in")

        # 编码器
        # 把encoder_inputs交给编码器，返回一个输出(预测序列的第一个值)和一个状态(传给解码器)
        (encoder_output_tensor, states) = tflearn.lstm(encoder_inputs, self.word_vec_dim, return_state=True, scope='encoder_lstm')
        encoder_output_sequence = tf.stack([encoder_output_tensor], axis=1)

        # 解码器
        # 预测过程用前一个时间序的输出作为下一个时间序的输入
        # 先用编码器的最后一个输出作为第一个输入
        if feed_previous:
            first_dec_input = go_inputs
        else:
            first_dec_input = tf.slice(decoder_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
        decoder_output_tensor = tflearn.lstm(first_dec_input, self.word_vec_dim, initial_state=states, return_seq=False, reuse=False, scope='decoder_lstm')
        decoder_output_sequence_single = tf.stack([decoder_output_tensor], axis=1)
        decoder_output_sequence_list = [decoder_output_tensor]
        # 再用解码器的输出作为下一个时序的输入
        for i in range(self.max_seq_len-1):
            if feed_previous:
                next_dec_input = decoder_output_sequence_single
            else:
                next_dec_input = tf.slice(decoder_inputs, [0, i+1, 0], [-1, 1, self.word_vec_dim])
            decoder_output_tensor = tflearn.lstm(next_dec_input, self.word_vec_dim, return_seq=False, reuse=True, scope='decoder_lstm')
            decoder_output_sequence_single = tf.stack([decoder_output_tensor], axis=1)
            decoder_output_sequence_list.append(decoder_output_tensor)

        decoder_output_sequence = tf.stack(decoder_output_sequence_list, axis=1)
        real_output_sequence = tf.concat([encoder_output_sequence, decoder_output_sequence], 1)

        net = tflearn.regression(real_output_sequence, optimizer='sgd', learning_rate=0.1, loss='mean_square')
        model = tflearn.DNN(net)
        return model

    '''
    train
      训练模型
      训练完毕后，将模型存储到model/model
      为减轻负担，这里将n_epoch减少。
      :returns : model
    '''


    def train(self):
        trainXY, trainY = self.generate_training_data()
        model = self.model(feed_previous=False)
        model.fit(trainXY, trainY, n_epoch=max_epoch_num, snapshot_epoch=False, batch_size=1)
        model.save('model/model')
        return model


    '''
    load
      加载训练模型进行测试
      :returns : model
    '''


    def load(self):
        model = self.model(feed_previous=True)
        model.load('model/model')
        return model

#------------------------------------------------------------------------------#


'''
__main__
Entry method
'''


if __name__ == '__main__':
    phrase = sys.argv[1]
    if 3 == len(sys.argv):
        my_seq2seq = MySeq2Seq(word_vec_dim=word_vec_dim, max_seq_len=max_seq_len, input_file=sys.argv[2])
    else:
        my_seq2seq = MySeq2Seq(word_vec_dim=word_vec_dim, max_seq_len=max_seq_len)
    if phrase == 'train':
        my_seq2seq.train()
    else:
        model = my_seq2seq.load()
        trainXY, trainY = my_seq2seq.generate_training_data()
        predict = model.predict(trainXY)
        for sample in predict:
            print "predict answer"
            for w in sample[1:]:
                (match_word, max_cos) = vector2word(w)
                #if vector_sqrtlen(w) < 1:
                #    break
                print match_word, max_cos, vector_sqrtlen(w)
