from ptlreg.apy.core.aobject.AObjectList import AObjectList
from ptlreg.apy.core.filepath.FilePath import *
import os
from ptlreg.apy.core.aobject.HasTags import HasTags

def _endswithany(fstring, extensions):
    for e in extensions:
        if(fstring.lower().endswith(e)):
            return True;
    return False;


class FilePathList(HasTags, AObjectList):
    ElementClass=FilePath;
    def __init__(self, filepaths=None, base_dir=None, **kwargs):
        self.base_dir = base_dir;
        if(isinstance(filepaths, (list, tuple, AObjectList))):
            if(isinstance(filepaths[0], HasFilePath)):
                super(FilePathList, self).__init__(filepaths, **kwargs);
                return;
                # return super(FilePathList, self).__init__(filepaths, **kwargs);

            else:
                fpaths = [];
                for f in filepaths:
                    fpaths.append(FilePath(f));
                super(FilePathList, self).__init__(fpaths, **kwargs);
                return;
                # return super(FilePathList, self).__init__(fpaths, **kwargs);
        else:
            super(FilePathList, self).__init__(filepaths, **kwargs);
            return;
            # return super(FilePathList, self).__init__(filepaths, **kwargs);

    def __str__(self):
        rstring = '';
        for a in self.paths:
            rstring=a+'\n'+rstring;
        return rstring;

    def show(self):
        print(self);

    @staticmethod
    def from_directory(directory, recursive=False, extension_list=None):
        new_list = FilePathList(base_dir=directory);
        if(not recursive):
            for filename in os.listdir(directory):
                if((extension_list is None) or _endswithany(filename, extension_list)):
                    new_list.append(FilePath(os.path.join(os.path.abspath(directory), filename)));
        else:
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if((extension_list is None) or _endswithany(f, extension_list)):
                        new_list.append(os.path.join(root, f));
        return new_list;

    @staticmethod
    def from_directory_search(directory, recursive=False, extension_list=None, criteriaFunc=None):
        """
        List of files in a directory that meet an optional criteriaFunc criteria function. If criteriaFunc(file_path)
        returns true then file_path will be included. If no criteriaFun is given, this function acts like FromDirectory
        :param directory:
        :param recursive:
        :param extension_list:
        :param criteriaFunc:
        :return:
        """
        new_list = FilePathList(base_dir=directory);
        if (not recursive):
            for filename in os.listdir(directory):
                if ((extension_list is None) or _endswithany(filename, extension_list)):
                    fpth = FilePath(os.path.join(os.path.abspath(directory), filename));
                    if (criteriaFunc):
                        if (criteriaFunc(fpth)):
                            new_list.append(fpth);
                    else:
                        new_list.append(fpth);
        else:
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if ((extension_list is None) or _endswithany(f, extension_list)):
                        fpth = FilePath(os.path.join(root, f));
                        if (criteriaFunc):
                            if (criteriaFunc(fpth)):
                                new_list.append(fpth);
                        else:
                            new_list.append(fpth);
        return new_list;

    @property
    def paths(self):
        def func(f):
            return f.test_file_path;
        return self.map(func);

    @property
    def file_names(self):
        def getfname(f):
            return f.file_name;
        return self.map(getfname);

    @property
    def file_name_base(self):
        def func(f):
            return f.file_name_base;
        return self.map(func);

    @property
    def directories(self):
        def func(f):
            return f.get_directory_path();
        return self.map(func);

    @property
    def extensions(self):
        def func(f):
            return f.file_ext;
        return self.map(func);

    @property
    def absolute_paths(self):
        def func(f):
            return f.absolute_file_path;
        return self.map(func);

    def relative_paths(self, from_path):
        def func(f):
            return os.path.relpath(f.absolute_file_path, from_path);
        return self.map(func);

    def relative_paths_to(self, to_path):
        def func(f):
            return os.path.relpath(to_path, f.absolute_file_path);
        return self.map(func);




