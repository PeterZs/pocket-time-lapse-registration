import os
import hashlib

from ptlreg.apydn import DataNodeConstants
from ptlreg.apydn.HasDataLabels import HasDataLabels
from ptlreg.apy.core.filepath.HasFilePath import HasFilePath

def _pathstring(path):
    return path.replace(os.sep+os.sep, os.sep);

def _myisdir(fpath):
    return os.path.isdir(fpath);

class HasFilePathLabel(HasFilePath):
    """

    """
    FILE_PATH_KEY = "file_path"
    # FILE_NAME_KEY = "file_name"

    def __init__(self, path=None, root_path=None, **kwargs):
        # self._file_path = None;
        super(HasFilePathLabel, self).__init__(**kwargs);
        self._init_path(path=path, root_path=root_path);


    def _init_path(self, path=None, root_path=None, **kwargs):
        if (path):
            adjusted_input_path = self._get_adjusted_input_file_path(input_file_path=path);
            self._set_file_path(file_path=adjusted_input_path, root_path=root_path, **kwargs);
            self._init_after_file_path_set();

    # <editor-fold desc="Property: 'file_path'">

    @property
    def file_path(self):
        return self.get_file_path();

    def get_file_path(self):
        """
        Override the version that uses get_info to get it from the labels
        :return:
        """
        return self.get_label(HasFilePathLabel.FILE_PATH_KEY);

    def _set_file_path(self, file_path=None, root_path=None, **kwargs):
        oldpath = None;
        if(self.file_path):
            oldpath = self.get_absolute_path();
        if(root_path):
            self._set_root_path(root_path);
        self.set_label(DataNodeConstants.FILE_PATH_KEY, os.path.normpath(file_path));
        if(oldpath):
            self.on_path_changed(old_path=oldpath);
    # </editor-fold>


    @property
    def root_path(self):
        return self.get_label("root_path");
    # </editor-fold>


    def set_root_path(self, root_path):
        """
        This method should be overridden by subclasses to set the root path.
        The reason is that some subclasses might handle this differently, so it should be set explicitly.
        :param root_path:
        :return:
        """
        raise NotImplementedError;

    def _set_root_path(self, value):
        self.set_label("root_path", value);

    # <editor-fold desc="Property: 'md5_string'">
    @property
    def _md5_string(self):
        return self.get_label("md5_string");

    @_md5_string.setter
    def _md5_string(self, value):
        self.set_label('md5_string', value);
    # </editor-fold>

    def get_absolute_path(self, from_root=None):
        """
        Get the absolute path of the file. By default, from the stored root, but this can be substituted with an argument
        :param from_root: optional root path to use instead of the stored root_path
        :return:
        """
        if(self.file_path is None):
            return None;
        root_path = from_root;
        if(root_path is None):
            root_path = self.root_path;
        if(root_path is not None):
            return os.path.abspath(os.path.join(self.root_path, self.file_path));
        else:
            return os.path.abspath(self.file_path);

    def on_path_changed(self, old_path=None, **kwargs):
        """
        Anything that should be done when _setFilePath changes the path.
        :param new_path:
        :param old_path:
        :param kwargs:
        :return:
        """
        # self._md5 = None;
        if (hasattr(super(HasFilePathLabel, self), 'on_path_changed')):
            return super(HasFilePathLabel, self).on_path_change(old_path=old_path, **kwargs);

    def file_exists(self, from_root=None):
        fpath = self.file_path;
        if(from_root is not None):
            fpath = os.path.join(from_root, fpath);
        return os.path.exists(fpath);

    def relative_path(self, to_path=None, from_root=None):
        fpath = self.get_absolute_path(from_root=from_root);
        if(to_path is None):
            return os.path.relpath(fpath);
        else:
            return os.path.relpath(fpath, to_path);

    def get_path_string_from_root(self, root_path):
        return os.path.join(root_path, self.file_path);
