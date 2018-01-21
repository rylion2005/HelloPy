#!/usr/bin/env python


import os
import sys
import argparse


#-----------------------------------------------------------------------------
    
#-----------------------------------------------------------------------------

def commandline(args=None, arglist=None):
    '''
    Main command line..
    '''

    help_text = """
Commands:

train - give size of training set to use, as argument
predict - give input sequence as argument (or specify inputs via --from-file <filename>)

"""
    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("cmd", help="command")
    parser.add_argument("cmd_input", nargs='*', help="input to command")
    parser.add_argument("-m", "--model", help="seq2seq model name: either embedding_rnn (default) or embedding_attention", default=None)
    parser.add_argument("-r", "--learning-rate", type=float, help="learning rate (default 0.0001)", default=0.0001)
    parser.add_argument("-e", "--epochs", type=int, help="number of trainig epochs", default=10)
    parser.add_argument("-i", "--input-weights", type=str, help="tflearn file with network weights to load", default=None)
    parser.add_argument("-o", "--output-weights", type=str, help="new tflearn file where network weights are to be saved", default=None)
    parser.add_argument("-p", "--pattern-name", type=str, help="name of pattern to use for sequence", default=None)
    parser.add_argument("-n", "--name", type=str, help="name of model, used when generating default weights filenames", default=None)
    parser.add_argument("--in-len", type=int, help="input sequence length (default 10)", default=None)
    parser.add_argument("--out-len", type=int, help="output sequence length (default 10)", default=None)
    parser.add_argument("--from-file", type=str, help="name of file to take input data sequences from (json format)", default=None)
    parser.add_argument("-L", "--num-layers", type=int, help="number of RNN layers to use in the model (default 1)", default=1)
    parser.add_argument("--cell-size", type=int, help="size of RNN cell to use (default 32)", default=32)
    parser.add_argument("--cell-type", type=str, help="type of RNN cell to use (default BasicLSTMCell)", default="BasicLSTMCell")
    parser.add_argument("--embedding-size", type=int, help="size of embedding to use (default 20)", default=20)
    parser.add_argument("--tensorboard-verbose", type=int, help="tensorboard verbosity level (default 0)", default=0)

    if not args:
        args = parser.parse_args(arglist)
    
    if args.iter_num is not None:
        print("iter_num %s" % args.iter_num)

    if args.cmd=="train":
        try:
            num_points = int(args.cmd_input[0])
        except:
            raise Exception("Please specify the number of datapoints to use for training, as the first argument")
        
    elif args.cmd=="predict":
        if args.from_file:
            inputs = json.loads(args.from_file)
        try:
            input_x = map(int, args.cmd_input)
            inputs = [input_x]
        except:
            raise Exception("Please provide a space-delimited input sequence as the argument")

    else:
        print("Unknown command %s" % args.cmd)

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------

if __name__=="__main__":
    commandline()
