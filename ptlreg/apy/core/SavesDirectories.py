from .SavesToJSON import SavesToJSON
from .filepath.FilePath import *
from .aobject.AObject import *

#import shutil
import os
from distutils.dir_util import copy_tree
import ptlreg.apy.utils
import shutil


def _pathstring(path):
    return path.replace(os.sep+os.sep, os.sep);

def _get_dir_from_path(pth):
    return (os.path.split(pth)[0]+os.sep);

class SavesDirectoriesMixin(object):
    """SavesDirectoriesMixin (class): Manages mediagraph. This should really be replaced with a database of some sort...
        Attributes:
    """

    # @staticmethod
    # def AObjectType():
    #     return 'SavesDirectoriesMixin';

    def __init__(self, path=None, **kwargs):
        """
        If you provide a directory, it will look for an existing SavesDirectoriesMixin.json in that directory, or create one if it does not already exist. If you provide the path to an existing json, it will use that json. If you provide a path to anything else it will complain...
        :param path:
        :param kwargs: if you include 'clear_temp=False' then it will not clear the temp directory upon creation
        """
        self._reset_directories();
        super(SavesDirectoriesMixin, self).__init__(path=path, **kwargs);
        if(kwargs.get('clear_temp', True)):
            self.clear_temp_dir();

    @property
    def managed_dir(self):
        return self.get_directory_path();

    def _reset_directories(self):
        self.directories = {};

    def _get_adjusted_input_file_path(self, input_file_path):
        """
        If provided an input file path, infers the actual file path to look for.
        Generally, if you provide a directory path, returns the path to the default JSON name inside that directory
        :param input_file_path:
        :return:
        """
        adjusted_path = input_file_path;
        if(adjusted_path is None):
            return None;
        fpath = FilePath(input_file_path);
        if(fpath.looks_like_dir):
            adjusted_path = os.path.join(adjusted_path, self.default_json_name());
        return adjusted_path;

    def _set_file_path(self, file_path=None, **kwargs):
        super(SavesDirectoriesMixin, self)._set_file_path(file_path=file_path, **kwargs);

    def _init_after_file_path_set(self, **kwargs):
        super(SavesDirectoriesMixin, self)._init_after_file_path_set(**kwargs);
        self.init_dirs();

    def init_dirs(self, **kwargs):
        self.add_dir_if_missing(name='misc', folder_name="Misc");
        if(self.get_dir('backup') is None):
            self.set_dir('backup', _pathstring(self.get_dir('misc', absolute_path=False) + "Backups" + os.sep));
        ptlreg.apy.utils.make_sure_dir_exists(self.get_dir('backup', absolute_path=True))
        if (self.get_dir('temp') is None):
            self.set_dir('temp', _pathstring(self.get_dir('misc', absolute_path=False) + "TEMP" + os.sep));
        ptlreg.apy.utils.make_sure_dir_exists(self.get_dir('temp', absolute_path=True))

    def relative_path(self, path):
        return os.path.relpath(path, start=self.managed_dir)

    def get_temp_dir(self, absolute_path = True):
        return self.get_dir('temp', absolute_path = absolute_path);

    def clear_temp_dir(self):
        self.empty_dir(name='temp');

    def on_path_change(self, new_path=None, old_path=None, **kwargs):
        """
        Anything that should happen when _set_file_path changes the path. For example, if we save an object to a json, move the json, and then load it, then the path will change from where the object was last saved to where the object was last loaded.
        :param new_path:
        :param old_path:
        :param kwargs:
        :return:
        """

        def prefix_removed(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix):]
            return text


        if(new_path is None or old_path is None):
            return super(SavesDirectoriesMixin, self).on_path_change(new_path=new_path, old_path=old_path, **kwargs);
        new_abs_path = os.path.abspath(new_path);
        old_abs_path = os.path.abspath(old_path);
        if(new_abs_path != old_abs_path):
            # print("Converting path from {} to {}".format(old_path, new_path));
            oldir = _get_dir_from_path(_pathstring(old_abs_path));
            newdir = _get_dir_from_path(_pathstring(new_abs_path));
            if (oldir != newdir):
                # AWARN("{} FOUND FILE MOVED FROM:\n{}\nTO:\n{}\nUPDATING DIRECTORIES...".format(self.aobject_type_name(), oldir, newdir));
                for d in self.directories:
                    dpth = self.directories[d];

                    if (os.path.isabs(dpth) and dpth.startswith(oldir)):
                        dpthst = prefix_removed(dpth, oldir);
                        self.directories[d] = os.path.join(newdir, dpthst);
                        AWARN("Directory path {} updated to {}".format(dpth, self.directories[d]));

        return super(SavesDirectoriesMixin, self).on_path_change(new_path=new_path, old_path=old_path, **kwargs);

    def set_dir(self, name, path):
        self.directories[name]=path;
        full_path = self.get_dir(name, absolute_path=True);
        ptlreg.apy.utils.make_sure_path_exists(full_path);
        return path;

    def add_dir(self, name, folder_name = None):
        assert(name not in self.directories), "tried to add {} dir to SavesDirectoriesMixin, but this dir is already set".format(name)
        if(folder_name is None):
            folder_name = name;
        return self.set_dir(name, _pathstring('.' + os.sep + folder_name + os.sep));
        # return self.setDir(name, pathstring(self.get_directory_path() + os.sep + folder_name + os.sep));

    def add_dir_if_missing(self, name, folder_name=None):
        if(name in self.directories):
            # if(not os.path.exists(self.get_dir(name))):
            ptlreg.apy.utils.make_sure_dir_exists(self.get_dir(name, absolute_path=True));
            return;
        else:
            self.add_dir(name=name, folder_name=folder_name);


    def add_subdir_if_missing(self, name, parent_name, folder_name):
        if (name in self.directories):
            ptlreg.apy.utils.make_sure_dir_exists(self.get_dir(name, absolute_path=True));
            return;
        else:
            self.set_dir(name,
                         _pathstring(self.get_dir(parent_name, absolute_path=False) + folder_name + os.sep));
            ptlreg.apy.utils.make_sure_dir_exists(self.get_dir(name, absolute_path=True));

    def get_dir(self, name, absolute_path = True):
        dir_path = self.directories.get(name);
        if(dir_path is None):
            return None;
        if(not os.path.isabs(dir_path) and absolute_path):
            dir_path = self._to_absolute_path(dir_path);
        return dir_path;

    def _to_absolute_path(self, path):
        """
        If path is relative, returns the absolute path assuming path is relative to the directorypath of this object. Is path is already absolute, it gets returned;
        :param path:
        :return:
        """
        path_is_relative = (path and not os.path.isabs(path));
        if(path_is_relative):
            base_dir = self.get_directory_path();
            if(base_dir is None):
                return None;
            apath = os.path.abspath(os.path.join(base_dir, path));
            if(path[-1] == os.sep):
                apath = apath+os.sep;
            return apath;
        else:
            return path;

    def empty_dir(self, name):
        dpth = self.get_dir(name, absolute_path=True);
        if(dpth is not None and os.path.isdir(dpth)):
            shutil.rmtree(dpth);
            ptlreg.apy.utils.make_sure_path_exists(dpth);

    def delete_all_dirs(self, really=False):
        if(not really):
            raise Exception("You need to REALLY want to delete all dirs...")
        for d in self.directories:
            dpth = self.get_dir(d, absolute_path=True);
            if (dpth is not None and os.path.isdir(dpth)):
                shutil.rmtree(dpth);
        self.directories = {};

    def empty_all_dirs(self, really=False):
        if (not really):
            raise Exception("You need to REALLY want to empty all dirs...")
        for d in self.directories:
            self.empty_dir(d);


    def delete_dir(self, name):
        dpth = self.get_dir(name, absolute_path=True);
        if (dpth is not None and os.path.isdir(dpth)):
            shutil.rmtree(dpth);
            d = dict(self.directories);
            del d[name];
            self.directories=d;

    def _DELETE_MANAGED_DIRECTORY(self, really=False):
        """
        Deletes the root of the SavesDirectoriesMixin object. This will delete all of the internal directories as well.
        :return:
        """
        if (not really):
            raise Exception("You need to REALLY want to delete the managed dir...")
        shutil.rmtree(self.get_directory_path());

    def to_dictionary(self):
        d = super(SavesDirectoriesMixin, self).to_dictionary();
        d['directories']=self.directories;
        #serialize class specific members
        return d;

    def copy_path_to_dir(self, path_to_copy, dest_dir):
        dest_path = self.get_dir(dest_dir, absolute_path=True);
        if(dest_path):
            if(os.path.isdir(path_to_copy)):
                copy_tree(src=path_to_copy, dst=dest_path);
            elif(os.path.isfile(path_to_copy)):
                shutil.copy2(path_to_copy, dest_path)
        return;

    def copy_dir_to_path(self, dir_to_copy, dest_path):
        src_path = self.get_dir(dir_to_copy, absolute_path=True);
        if(src_path):
            if(os.path.isdir(dest_path)):
                copy_tree(src=src_path, dst=dest_path);
        return;

    def init_from_dictionary(self, d):
        self._reset_directories();
        super(SavesDirectoriesMixin, self).init_from_dictionary(d);
        self.directories = d['directories'];



    def save_json(self, json_path=None, on_file_exists='backup', **kwargs):
        save_path = json_path;
        if(save_path is None):
            save_path = self.get_file_path();
        assert(save_path.lower().endswith(self._CLASS_JSON_FILE_EXTENSION())), "Directory info saves to json: cannot save to path '{}'".format(save_path);
        if (os.path.isfile(save_path)):
            if(on_file_exists.lower() == 'fail'):
                assert(False), "File {} already exists.".format(save_path);
            elif (on_file_exists.lower() == 'replace'):
                AWARN("Replacing file {}".format(save_path));
            else:
                os.rename(save_path, self.get_dir('backup', absolute_path=True) + os.sep + os.path.basename(save_path));

        return super(SavesDirectoriesMixin, self).save_json(json_path=save_path, **kwargs)
        # self.writeToJSON(save_path);


class SavesDirectories(SavesDirectoriesMixin, SavesToJSON, AObject):
    ALLOW_DIRECTORIES = True
    pass;
