#!/usr/bin/env python

# -*- coding: UTF-8 -*-

import getopt
import os
import sys
import time
import commands
import thread
import zipfile
import shutil


USAGE = '''
version : 0.1.0
   date : 2017-12-12
authtor : Julien Ryan

usage: python [option] ... [-h|-i|-o|-w|-e|-r|-b|-s|-d] [arg] ...

Options and arguments (and corresponding environment variables):
-h help         : this help
-i <input file> : input file
-o <out file>   : output file
-s <device>     : directs command to the device or emulator with the given
                  serial number or qualifier. Overrides ANDROID_SERIAL 
                  environment variable.
-e              : extract crash logs as zip file
-r              : check radio properties information
-b              : capture bootup log
-c <command>    : directly execute a adb command looply
-d <apkfile>    : decompile apk file
-w              : write wifi ssid/key directly to wpa_supplicant.conf
                  --ssid and --key MUST be input together. 
--ssid <ssid>   : wifi ssid
--key   <key>   : wifi password, length is more than 8-digits
--audio         : dump audio information
   

'''


short_options = 'hwerbi:o:s:c:d:'
long_options = ['help', 'input=', 'output=', 'ssid=', 'key=', 'audio']
now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
input_file = ''
output_file = ''
device = ''


'''
usage
print command usage
'''


def usage():
    print USAGE


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
Capture boot up log

:synchronized boolean: if capture log synchronize
:returns: None
:raises KeyError: None 
'''


def capture_boot(synchronized):
    print 'capture boot: ', synchronized
    boot_log_file = 'boot-' + now + '.txt'
    boot_log_cmd = 'adb logcat -b all -v threadtime > ' + boot_log_file
    poll_device()
    if synchronized:
        print 'printing log ......'
        os.popen(boot_log_cmd)
    else:
        print 'printing log in the other thread'
        thread.start_new(os.system, (boot_log_cmd,))
        print 'wait 60 seconds ......'
        time.sleep(60)


'''
loop_command

:command, str: command to be executed
:returns: None
:raises KeyError: None
'''


def loop_command(command):
    print 'loop command: ', command
    poll_device()
    while True:
        os.popen(command)


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


'''
write_wifi
write wifi directly to wpa_supplicant.conf
:ssid, str: ssid
:key, str:  key
:returns: None
:raises KeyError: None
'''


def write_wifi(ssid, key):
    ssid_key = '''
        network={
            ssid="FAKESSID"
            scan_ssid=1
            psk="FAKEPASSWORD"
            proto=RSN
            key_mgmt=WPA-PSK
            group=CCMP TKIP
            sim_num=1
        }
        '''
    print 'write wifi ssid/key'
    ssid_key.replace('FAKESSID', ssid).replace('FAKEPASSWORD', key)

    poll_device()

    result = commands.getoutput('adb pull /data/misc/wifi/wpa_supplicant.conf .')
    if result.find('not exist') == -1:
        print 'wifi config file found !'
        f = open('wpa_supplicant.conf', 'a')
        f.write(ssid_key)
        f.close()

        print 'push back to phone'
        os.system('adb push wpa_supplicant.conf /data/misc/wifi/')
        os.system('adb shell chown wifi:wifi /data/misc/wifi/wpa_supplicant.conf')
        os.system('adb shell chmod 660 /data/misc/wifi/wpa_supplicant.conf')

        os.remove('wpa_supplicant.conf')
    else:
        print '\n!!!!!!!!!!  Please check if you have open wifi !!!!!!!!!\n'
        sys.exit(-1)


'''
check_radio
check radio properties
'''


def check_radio():
    print 'check radio'
    radio_info_file = 'radio_prop.txt'
    radio_command = 'adb shell getprop |grep -iE \"gsm|radio|telephony|ril\"' + ' > ' + radio_info_file
    os.system(radio_command)


'''
dump_audio
dump audio information
'''


def dump_audio(outfile):
    print 'dump audio'

    audiodump = outfile
    commandss = (
        'adb shell dumpsys media_session',
        'adb shell dumpsys media_router',
        'adb shell dumpsys media_projection',
        'adb shell dumpsys media.player',
        'adb shell dumpsys media.audio_flinger',
        'adb shell dumpsys media.sound_trigger_hw',
        'adb shell dumpsys audio',
        'adb shell dumpsys audio_policy',
        'adb shell ps')

    poll_device()

    if (audiodump is None) or (len(audiodump) == 0):
        audiodump = 'audiodumps.txt'

    print 'dump file: ' + audiodump
    if os.path.isfile(audiodump):
        os.remove(audiodump)

    for cmmd in commandss:
        print cmmd
        os.popen('echo ' + '"\n==========\n"' + cmmd + ' >> ' + audiodump)
        os.popen(cmmd + ' >> ' + audiodump)


'''
decompile_apk
decompile apk file
'''


def decompile_apk(apkfile):
    print 'decompile_apk: ' + apkfile
    if not os.path.exists(apkfile):
        print 'apk file not exist'
        sys.exit(0) 
    elif not os.path.exists('apktools/apktool_2.3.0.jar'):
        print 'apk tool not exist'
        sys.exit(0)
    elif not os.path.exists('apktools/jd-gui-1.4.0.jar'):
        print 'jd-gui not exist'
        sys.exit(0)
    elif not os.path.isdir('apktools/dex2jar-2.0'):
        print 'dex2jar not exist'
        sys.exit(0)
    else:
        print 'everything is ok'

    tmpzipfile = apkfile.replace('apk', 'zip')
    shutil.copyfile(apkfile, tmpzipfile)
    outdir = apkfile + '_FILES'
    unzip_file(tmpzipfile, outdir)

    dexsrc = outdir + '/classes.dex'
    dex2jar = './apktools/dex2jar-2.0/d2j-dex2jar.sh -f ' + dexsrc
    os.popen(dex2jar)

    os.remove(tmpzipfile)


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


'''
init_args
initialize the command line arguments
'''


def init_args():
    global device
    global input_file
    global output_file

    if len(sys.argv) <= 1:
        usage()
        sys.exit(0)

    write_wifi_request = False
    ssid = ''
    key = ''

    audio_dump_request = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.GetoptError:
        print '\nsorry, i don\'t know you !\n'
        usage()
        sys.exit(-1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-o", "--output"):
            output_file = arg
        elif opt == '-s':
            device = arg
        elif opt == '-e':
            extract_crash_logs()
        elif opt == '-r':
            check_radio()
        elif opt == '-b':
            capture_boot(True)
        elif opt == '-c':
            loop_command(arg)
        elif opt == '-w':
            write_wifi_request = True
        elif opt == '-d':
            decompile_apk(arg)
        elif opt == '--ssid':
            ssid = arg
        elif opt == '--key':
            key = arg
        elif opt == '--audio':
            audio_dump_request = True;
        else:
            print '\nWho are you?\n'
            usage()

    if write_wifi_request is True:
        if ssid == '':
            print 'ssid is MUST !'
            sys.exit(-1)
        elif key == '':
            print 'key is MUST !'
            sys.exit(-1)
        else:
            write_wifi(ssid, key)

    if audio_dump_request is True:
        dump_audio(output_file)


'''
    main entry
'''
if __name__ == "__main__":
    init_args()
    print '\n!!!!!! All finished !!!!!!\n'
