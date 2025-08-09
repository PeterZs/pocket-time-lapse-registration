#AImports
from future.utils import iteritems
import os
import os.path
import errno
import json
try:
    import cPickle as pickle
except:
    import pickle
from time import gmtime, strftime, localtime
import random
import uuid
import string
import unicodedata
import hashlib
import sys

import ptlreg.apy.defines

import jsonpickle
jsonpickle.set_encoder_options('simplejson', sort_keys = True, indent = 4, ensure_ascii=False);
jsonpickle.set_encoder_options('json', sort_keys = True, indent = 4, ensure_ascii=False);
import itertools
import platform
import ptlreg.apy.defines as apydefines


try:
    import ptlreg.apy.afileui;
    apydefines.HAS_FILEUI = True;
    if(platform.system() == 'Darwin'):
        apydefines.HAS_FILEUI = True;
    # print("Running on a Mac with platform.release()={}".format(platform.release()));
except ImportError:
    apydefines.HAS_FILEUI = False;

try:
    from termcolor import colored
    def AWARN(message):
        if (apydefines.AD_DEBUG):
            print(colored(message, 'red'))
    def AINFORM(message):
        if(apydefines.AD_DEBUG):
            print(colored(message, 'blue'))
except ImportError:
    if (apydefines.AD_DEBUG):
        print("You do not have termcolor installed (pip install termcolor). AWARN will just show as plain print statements when apydefines.AD_DEBUG==True...")
    def AWARN(message):
        if (apydefines.AD_DEBUG):
            print(message);
    def AINFORM(message):
        if (apydefines.AD_DEBUG):
            print(message);

# Python 3 compatibility hack
try:
    unicode('');
except NameError:
    unicode = str;

asunicode = unicode;

def __current_running_python_version__():
    if ((3, 0) <= sys.version_info):
        return 3;
    elif ((2, 0) <= sys.version_info):
        return 2;
    else:
        assert(False), 'AOBLITE DOES NOT WORK WITH PYTHON VERSION: {}'.format(sys.version_info);

def GetUUID():
    return uuid.uuid1();

def aget_ipython():
    try:
        import IPython
        return IPython;
    except ImportError:
        AWARN('Problem importing IPython');
        return None;

def GetTempDir():
    return apydefines.TEMP_DIR;

def SetTempDir(value):
    apydefines.TEMP_DIR = value;


def runningInNotebook():
    try:
        ipyth = aget_ipython();
        if(ipyth.__class__.__name__ == 'module'):
            ipyth = ipyth.get_ipython();
        shell = ipyth.__class__.__name__;

        # shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def runningInSpyder():
    return 'SpyderKernel' in str(aget_ipython().kernel.__class__);


###############

def getPythonKernelID():
    from IPython.lib import kernel
    connection_file_path = kernel.get_connection_file()
    connection_file = os.path.basename(connection_file_path)
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]
    return kernel_id;

_ISNOTEBOOK = False;
if(runningInNotebook()):
    _ISNOTEBOOK = True;

def is_notebook():
    return _ISNOTEBOOK;
    # return runningInNotebook()

def subdict(subkeys, full_dict):
    return dict((k, full_dict[k]) for k in subkeys if k in full_dict);

def suppress_outputs():
    return (is_notebook() and apydefines.SUPPRESS_OUTPUTS);

def local_time_string():
    return strftime(apydefines.TIMESTAMP_FORMAT, localtime());


def cmp_to_key(mycmp):
    """
    Convert a cmp= function into a key= function
    e.g.:
    sorted(list_object, key=cmp_to_key(cmp_func))
    :param mycmp:
    :return:
    """
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K




def get_temp_file_path(final_file_path="TEMP", temp_dir_path = None):
    """

    :param final_file_path: If you want the path for an intermediate file you can provide the final file name and the
    temp file will be made to look similar
    :param temp_dir_path: The directory where the temp file will go
    :return:
    """
    pparts = os.path.split(final_file_path);
    destfolder = pparts[0]+os.sep;
    tempdir = temp_dir_path;
    if(tempdir is None):
        tempdir=GetTempDir();
        # tempdir='.';
    destfolder=pathstring(tempdir+os.sep);
    tempname = 'TEMP_'+pparts[1];
    temptry = 0;
    while(os.path.isfile(destfolder+tempname)):
        temptry=temptry+1;
        tempname = 'TEMP{}_'.format(temptry)+pparts[1];
    return pathstring(destfolder+tempname);


def get_incremented_until_path_does_not_exist(initial_path_name):
    namesplit = os.path.splitext(initial_path_name);
    newpathname = initial_path_name;
    nametry = 0;
    while(os.path.exists(newpathname)):
        nametry = nametry+1;
        newpathname = '{}_{}'.format(namesplit[0], nametry)+namesplit[1];
    return newpathname;

# def dealiasargs(in_args, pass_arg, **kwargs):
#     """
#     example use:
#     dealiasargs(in_args=['path', 'address'], pass_arg='path', path=None, address='here')
#     will return a dictionary {'path':'here'}
#     dealiasargs(in_args=['path', 'address'], pass_arg='path', path='there', address=None)
#     will return a dictionary {'path':'there'}
#     dealiasargs(in_args=['path', 'address'], pass_arg='path', path='there', address='here')
#     will assert
#     :param pass_argument:
#     :param kwargs:
#     :return:
#     """
#     rval = None;
#     rkey = None;
#     for a in in_args:
#         if(a in kwargs.keys()):
#             assert(rval is None), "Cannot provide both {} and {} arguments".format(a, rkey);
#             rval = kwargs[a];
#             if(rval is not None):
#                 rkey = a;
#     #     print("rkey: {}\nrval:{}".format(rkey, rval))
#     if(rkey == pass_arg):
#         return kwargs;
#     else:
#         rdict = dict(kwargs);
#         for ia in in_args:
#             if(ia != pass_arg):
#                 del rdict[ia];
#         rdict[pass_arg]=rval;
#         return rdict;

# print(dealiasargs(in_args=['managed_dir', 'path'],
#                   path=None,
#                   pass_arg='path',
#                   managed_dir='hello there',
#                   **{'thisthing': 'thisvalue'}))

#

def getShellName():
    try:
        shell = aget_ipython().__class__.__name__
        return shell;
    except NameError:
        return False      # Probably standard Python interpreter

def md5string(file_path):
    return hashlib.md5(file_path).hexdigest()


def pickleToPath(d, path):
    f = open(path, 'wb');
    pickle.dump(d, f, protocol=2);
    f.close();
    return True;

def unpickleFromPath(path):
    f=open(path, 'rb');
    d=pickle.load(f);
    f.close();
    return d;

def make_sure_path_exists(path):
    """

    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def make_sure_dir_exists(path):
    """

    :param path:
    :return:
    """
    pparts = os.path.split(path); #Does this return bytes?
    destfolder = pparts[0]+os.sep;
    try:
        os.makedirs(destfolder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def file_is_inside_directory(file_path, directory_path):
    return os.path.abspath(file_path).startswith(os.path.abspath(directory_path));

def same_file_path(path1, path2):
    return os.path.abspath(path1)==os.path.abspath(path2);

def path_looks_like_dir(path):
    return (path == os.path.splitext(path)[0]);

def safe_file_name(input_string):
    return ''.join([i if ord(i) < 128 else '_' for i in input_string]);

def pathstring(path):
    return path.replace(os.sep+os.sep, os.sep);

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

# def printAttributes(obj):
#     for attr in dir(obj):
#         print("#### Obj.%s = %s\n" % (attr, getattr(obj, attr))

def get_prepended_name_file_path(original_file_path, string_to_prepend):
    pparts = os.path.split(original_file_path);
    destfolder = pparts[0]+os.sep;
    pname = string_to_prepend+pparts[1];
    return pathstring(destfolder+pname);

def change_extension(input_path, new_ext):
    nameparts = os.path.splitext(input_path);
    return nameparts[0]+new_ext;

def printDictionary(obj):
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                printDictionary(v)
            else:
                print("{}: {}\n".format(k, v))
                # print '%s : %s' % (k, v)
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                printDictionary(v)
            else:
                print(v)
    else:
        print(obj);

def N_Nones(*args):
    return len([x for x in args if x is None]);

def N_NonNones(*args):
    return len([x for x in args if x is not None]);

def cartesian_product(*args, **kwargs):
    """
    example:
    cartesian_product([-1,1],[-1,1])= [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    :param args:
    :param kwargs:
    :return:
    """
    return list(itertools.product(*args, **kwargs));

# def formatdictionary(obj):
#     rstr = '';
#     def printst(a, rstr):
#         return rstr+a+'\n';
#     if type(obj) == dict:
#         for k, v in obj.items():
#             if hasattr(v, '__iter__'):
#                 rstr = printst(k, rstr)
#                 rstr = printst(formatdictionary(v), rstr)
#             else:
#                 ks = k;
#                 if(ks is None):
#                     ks = 'None';
#                 vs = v;
#                 if(vs is None):
#                     vs = 'None';
#                 print(k)
#                 print(v is None)
#                 rstr = printst("{} : {}".format(ks,vs), rstr)
#     elif type(obj) == list:
#         for v in obj:
#             if hasattr(v, '__iter__'):
#                 rstr = printst(formatdictionary(v), rstr)
#             else:
#                 rstr = printst(v, rstr)
#     else:
#         rstr = printst(obj, rstr)
# def spotgt_shift_bit_length(x):
#     #smallest power of two greater than
#     assert(isinstance(x,int)), "In 'spotgt_shift_bit_length(x)' x must be integer."
#     return 1<<(x-1).bit_length()

def get_file_name_from_path(pth):
    return os.path.split(pth)[1];

def get_dir_from_path(pth):
    return (os.path.split(pth)[0]+os.sep);

def get_file_names_from_paths(pths):
    r = [];
    for p in pths:
        r.append(get_file_name_from_path(p));
    return r;

def writeDictionaryToJSON(d, json_path=None):
    if(json_path):
        with open(json_path, 'w') as outfile:
            json.dump(d, outfile, sort_keys = True, indent = 4, ensure_ascii=False);


def vtt_to_srt(fileContents):
    replacement = re.sub(r'([\d]+)\.([\d]+)', r'\1,\2', fileContents);
    replacement = re.sub(r'WEBVTT\n\n', '', replacement);
    replacement = re.sub(r'^\d+\n', '', replacement);
    replacement = re.sub(r'\n\d+\n', '\n', replacement);
    return replacement;


# def increment_path_until_does_not_exist(file_path):
#     if(not os.path.exists(file_path)):
#         return file_path;
#     pparts = os.path.split(final_file_path);
#     destfolder = pparts[0] + os.sep;
#     tempdir = temp_dir_path;
#     if (tempdir is None):
#         tempdir = '.';
#     destfolder = pathstring(tempdir + os.sep);
#     tempname = 'TEMP_' + pparts[1];
#     temptry = 0;
#     while (os.path.isfile(destfolder + tempname)):
#         temptry = temptry + 1;
#         tempname = 'TEMP{}_'.format(temptry) + pparts[1];
#     return pathstring(destfolder + tempname);

def find_all_files_with_name_under_path(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def jsonsafedict(d):
    def safeit(a):
        if isinstance(a, unicode):
            return unicodedata.normalize('NFKD', a).encode('ASCII', 'ignore').decode('ASCII');
        else:
            return a;

    return {k: safeit(v) for k, v in iteritems(d)};
    # return {k: safeit(v) for k, v in d.iteritems()};

def nums2safestring(nums):
    sout = '';
    for n in nums:
        sout = sout + str(n) + 'v'
    return sout.replace('.', '_');

def random_letters(n_letters = 1):
    string.letters
    rletter = random.choice(string.letters);
    if(n_letters>1):
        for n in range(n_letters-1):
            rletter = rletter+random.choice(string.letters);
    return rletter;