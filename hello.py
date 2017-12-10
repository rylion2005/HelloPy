#!/usr/bin/env python

# -*- coding: UTF-8 -*-

import sys
import getopt
from optparse import OptionParser


USAGE = '''
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


"""
main

This is a reST style.

:param None: this is a first none param
:returns: none
:raises keyError: none
"""


def init_args():

    if len(sys.argv) <= 1:
        usage()
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "i:o:h", ["help", "input=", "output="])
        except getopt.GetoptError:
            print '\nsorry, i don\'t know you !\n'
            usage()
            sys.exit()

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                usage()
                sys.exit()
            elif opt in ("-i", "--input"):
                print arg
            elif opt in ("-o", "--output"):
                print arg
            else:
                print '\nWho are you?\n'
                usage()


def init_args0():
    try:
        parser = OptionParser()

        parser.add_option(
            "-i", "--input",
            action="store",
            dest="input",
            type=str,
            default="",
            help="input directory"
        )

        parser.add_option(
            "-o", "--output",
            action="store",
            dest="output",
            type=str,
            default="",
            help="output directory"
        )

        (options, args) = parser.parse_args()
        if options.input is not None:
            print "input"
            print args
        elif options.output is not None:
            print "output"
            print args
        else:
            parser.print_help()
    except Exception as ex:
        print("exception :{0}".format(str(ex)))
        print args


'''
    main entry
'''
if __name__ == "__main__":
    init_args()
