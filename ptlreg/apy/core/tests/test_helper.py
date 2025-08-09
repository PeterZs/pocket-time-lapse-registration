import shutil
import os
from ptlreg.apy.core import *
from ptlreg.apy.core import TestAObjects
from ptlreg.apy.core import TestAObjectLists
from ptlreg.apy.core.tests.TestClasses import *
import numpy as np
import random


testvals = dict(testint=1234567,
                testfloat = 3.1415927,
                teststring = 'Hello World!',
                testlist = [1, 'hello', 5.4321, {'innerdictkey':'innerdictvalue'}],
                testdict = dict(a=1,b=2,c=3));




def make_sure_dir_exists(path):
    pparts = os.path.split(path);
    destfolder = pparts[0]+os.sep;
    try:
        os.makedirs(destfolder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def GetTestDir(test_file_path):
    dir_path = os.path.dirname(os.path.realpath(test_file_path))
    dirname = os.path.splitext(os.path.basename(test_file_path))[0]+'_tempdir';
    counter = 0;
    testdir = os.path.join(dir_path,'tmp',dirname);
    while(os.path.isdir(testdir)):
        counter = counter+1;
        testdir = os.path.join(dir_path, 'tmp', dirname + str(counter));

    if(testdir[-1] !=os.sep):
        testdir = testdir+os.sep;
    return testdir;



