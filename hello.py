#!/usr/bin/env python

# -*- coding: UTF-8 -*-


"""
This is a reST style.

:param param1: this is a first param
:param param2: this is a second param
:returns: this is a description of what is returned
:raises keyError: raises an exception
"""

import sys
import getopt


USAGE = \
    '''
        usage: python [option] ... [-i input | -o output ] [arg] ...
        Options and arguments (and corresponding environment variables):
        -i input  : input directory
        -o output : output directory
        -h help   : this help
    '''


"""
    Usage
"""


def usage():
    print USAGE


'''
    main function
'''


def main(argv):
    if argv is None:
        argv = sys.argv
        print usage()
    else:
        opts, args = getopt.getopt(argv, "hi:o:", ["help", "input=", "output="])
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print usage()
                sys.exit()
            elif opt in ("-i", "--input"):
                print arg
            elif opt in ("-o", "--output"):
                print arg
            else:
                print 'over'
        print 'for over !'


'''
    main entry
'''
if __name__ == "__main__":
    # print sys.argv
    main(sys.argv[1:])