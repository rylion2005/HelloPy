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
load_vectors
加载词向量文件中的词向量

这个方法填充词向量字典vector_dictionary
    key = 单词
  value = 200维向量

'''


def load_vectors():
    print "begin load vectors " + vector_file
    f = open(vector_file, "rb")
    # 获取词表数目及向量维度
    words_and_size = f.readline()
    words_and_size = words_and_size.strip()
    words = long(words_and_size.split(' ')[0])
    size = long(words_and_size.split(' ')[1])
    print "words =", words
    print "size =", size

    for b in range(0, words):
        a = 0
        word = ''
        # 读取一个词
        while True:
            c = f.read(1)
            word = word + c
            if False == c or c == ' ':
                break
            if a < max_w and c != '\n':
                a = a + 1
        word = word.strip()

        vector = []
        for index in range(0, size):
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
        load_vectors()
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
