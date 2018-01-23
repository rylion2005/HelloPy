#!/usr/bin/env python


import os
import sys
import argparse
import getopt
import time
import commands
import thread
import zipfile
import shutil

#-----------------------------------------------------------------------------


'''
extract_crash_logs
extract crash logs from phone
'''


def extract_crash_logs():
    print 'extract crash logs'
    crash_dir = 'crashlogs'
    crash_zip = crash_dir + '-' + now + '.zip'
    command_arrary = (
        'adb pull /data/tombstones/ crashlogs/data/tombstones/',
        'adb pull /data/anr/ crashlogs/data/anr/',
        'adb pull /data/ramdump/ crashlogs/data/ramdump/',
        'adb pull /sdcard/ramdump/ crashlogs/sdcard/ramdump/',
        'adb pull /data/misc/SelfHost/RIDLCrash.txt crashlogs/data/misc/SelfHost/RIDLCrash.txt',
        'adb pull /data/system/dropbox/ crashlogs/data/system/dropbox/',
        'adb shell dmesg > crashlogs/dmesg.txt',
        'adb bugreport > crashlogs/bugreport.txt')

    poll_device()
    clear_log(crash_dir)
    for cmmd in command_arrary:
        print cmmd
        os.popen(cmmd)
    print 'extract over !'
    zip_dir(crash_dir, crash_zip)
    clear_log(crash_dir)


#-----------------------------------------------------------------------------


'''
poll_device
poll adb devices
'''


def poll_device():
    print now
    print 'polling device'
    while True:
        serial = commands.getoutput('adb get-serialno')
        if serial.find('unknown') != -1:
            continue
        else:
            os.system('adb root')
            time.sleep(2)
            os.system('adb remount')
            time.sleep(1)
            print 'device found !'
            break


'''
clear_log
remove log directory or files

:log, str: log, file or path
:returns: None
:raises KeyError: None 
'''


def clear_log(log):
    print 'clear old log'
    if os.path.isdir(log):
        shutil.rmtree(log)
    elif os.path.isfile(log):
        os.remove(log)
    else:
        print 'nothing, go next ...'


'''
zip dir
zip dir as one zip file

:log_dir, str: log directory with full path
:returns: None
:raises KeyError: None 
'''


def zip_dir(indir, outzip):
    print 'zip directory: ', indir
    zf = zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED)
    file_lists = []
    list_zip_files(indir, file_lists)
    for f in file_lists:
        print f
        zf.write(f)
    zf.close()


'''
list_zip_files
list the directory recursively

:input_path, str: input directory
:full_path_file str: output file with path
:returns: None
:raises KeyError: None
'''


def list_zip_files(input_path, out_path):
    files = os.listdir(input_path)
    for f in files:
        if os.path.isdir(input_path + '/' + f):
            list_zip_files(input_path + '/' + f, out_path)
        else:
            out_path.append(input_path + '/' + f)


'''
unzip_file
unzip zip file
'''


def unzip_file(zipped_file, out_dir):
    print 'unzip file: ' + zipped_file
    if not os.path.isfile(zipped_file):
        print zipped_file + 'not exists !'
        sys.exit(0)
    zf = zipfile.ZipFile(zipped_file, 'r')
    zf.extractall(out_dir)
    zf.close()


if __name__=="__main__":
    extract_crash_logs()
