
1. 电影对话语料收集整理
   分解成Q/A格式

2. 文本文件转换为utf-8

3. 切词
   切词成空格分行
   python segger.py xiaohuangji.txt xhj.seg


4. 生成词向量
   ./word2vec -train xhj.seg -output xhjvec.bin -size 200 -window 8 -sample 1e-4 -threads 20 -binary 0 -cbow 1 -iter 10

   注意啊，小样本，需要设置参数 -min-count 1; 大样本默认为5

5. 修改训练模型
   lstm_train.py
   my_seq2seq.py

6. 模型训练
   python my_seq2seq.py train
   训练完毕，生成./model/model模型文件

7. 结果文件测试
   python my_seq2seq_v2.py test test.data

8. 移植到Android工程

