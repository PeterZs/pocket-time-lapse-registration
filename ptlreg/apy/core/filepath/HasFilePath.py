
import os
import ptlreg.apy.defines
import ptlreg.apy.utils
import hashlib
import warnings
from datetime import datetime
import platform
from ptlreg.apy.defines import TIMESTAMP_FORMAT


try:
    from pathlib import Path as PathLibPath
    _HAVE_PATHLIB = True
    def PathOb(path):
        if (isinstance(path, PathLibPath)):
            return path
        else:
            return PathLibPath(path)
except ImportError:
    def PathOb(path):
        raise NotImplementedError;
        return path;
    _HAVE_PATHLIB = False

def datetime_from_formatted_timestamp_string(s):
    formats = [
        "%Y-%m-%dT%H-%M-%S",
        "%Y-%m-%d-%H-%M-%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt
        except ValueError:
            continue
    try:
        # Try ISO format
        dt = datetime.fromisoformat(s)
        return dt
    except Exception:
        return None

def get_file_creation_datetime(path_to_file, force_use_metadata=False):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if (not force_use_metadata):
        original_file_name = os.path.basename(path_to_file);
        original_file_name_base = os.path.splitext(original_file_name)[0];
        name_timestamp = datetime_from_formatted_timestamp_string(original_file_name_base);
        if(name_timestamp is not None):
            return name_timestamp;

        # if (name_timestamp is not None):
        #     dt = datetime.fromtimestamp(name_timestamp)
        #     return dt.strftime(TIMESTAMP_FORMAT)


    if platform.system() == 'Windows':
        return datetime.fromtimestamp(os.path.getctime(path_to_file))
    else:
        stat = os.stat(path_to_file)
        try:
            return datetime.fromtimestamp(stat.st_birthtime)
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return datetime.fromtimestamp(stat.st_mtime)

def file_created_timestamp_string_from_path(path, force_use_metadata=False):
    """
    Get the creation timestamp of a file as a formatted string. If the filename is formatted with a timestamp,
    that timestamp will be used instead of the file creation time, unless `force_use_metadata` is True.
    :param path:
    :param force_use_metadata:
    :return:
    """
    dt = get_file_creation_datetime(path, force_use_metadata=force_use_metadata)
    return dt.strftime(TIMESTAMP_FORMAT)

# def datetime_from_formatted_timestamp_string(formatted_str):
#     """
#     Converts a formatted string (YYYY-MM-DD HH:MM:SS) to a Unix timestamp (float).
#     """
#     dt = datetime.datetime.strptime(formatted_str, "%Y-%m-%d %H:%M:%S")
#     return dt.timestamp()


def _pathstring(path):
    return path.replace(os.sep+os.sep, os.sep);

def _myisdir(fpath):
    return os.path.isdir(fpath);
    # fpparts = os.path.splitext(fpath);
    # return (fpparts[1] is '');

class HasFilePath(object):
    """

    """
    ALLOW_DIRECTORIES = False

    def __init__(self, path=None, **kwargs):
        # self._file_path = None;
        super(HasFilePath, self).__init__(**kwargs)
        if(path is not None):
            self._init_path(path=path);

    def _init_path(self, path=None, **kwargs):
        adjusted_input_path = self._get_adjusted_input_file_path(input_file_path=path);
        # self._checkForPathChange(adjusted_input_path);
        self.set_file_path(file_path=adjusted_input_path, **kwargs);
        self._init_after_file_path_set();


    def _get_adjusted_input_file_path(self, input_file_path):
        return input_file_path;

    # <editor-fold desc="Property: 'file_path'">
    @property
    def file_path(self):
        return self.get_file_path();
    def get_file_path(self):
        return self.get_info('file_path');

    # @file_path.setter
    # def file_path(self, value):
    #     self._set_file_path(value);


    def _get_old_path(self):
        """
        later subclasses may replace this with get_absolute_path
        :return:
        """
        return self.file_path;

    def set_file_path(self, file_path=None, **kwargs):
        file_path_string = file_path;
        if (isinstance(file_path_string, HasFilePath)):
            file_path_string = file_path_string.get_absolute_path();

        oldpath = None;
        if (self.file_path):
            oldpath = self._get_old_path();
        oldpath = self.file_path;  # wherever the filepath thought it was before. For example, it may be that this was the file path when something was previously saved, but it has been moved and is now being loaded from the new location.
        # assert(fpparts[1] is not ''), "file_path {} looks like a directory, not a file".format(file_path);
        # assert(not _myisdir(file_path)), "file_path {} is a directory, not a file".format(file_path);
        if (file_path_string is not None):
            file_path_string = os.path.normpath(file_path_string)
            if (_myisdir(file_path_string) and (not self.__class__.ALLOW_DIRECTORIES)):
                warnings.warn("file_path {} for class {} is a directory, not a file".format(file_path_string,
                                                                                            self.__class__.__name__));

        self._set_file_path(file_path_string)
        self.on_path_change(new_path=self.file_path, old_path=oldpath);

    def _set_file_path(self, file_path=None, **kwargs):
        """
        Override this method to set the file path.
        :param file_path:
        :param kwargs:
        :return:
        """
        if(file_path is None):
            file_path = self.file_path;
        # if(file_path is None):
        #     raise ValueError("file_path cannot be None");
        self.set_info('file_path', file_path);
    # </editor-fold>


    def get_file_pathlib(self):
        return PathLibPath(self.file_path);

    @property
    def pathlib_path(self):
        return self.get_file_pathlib();

    def file_path_parts(self):
        '''

        Returns: a list of the path parts

        '''
        parts = [];
        split = os.path.split(self.get_absolute_path());
        while(split[1]!=''):
            parts.append(split[1]);
            split = os.path.split(split[0]);
        parts.append(split[0]);
        parts.reverse()
        return parts;

    def _init_after_file_path_set(self):
        pass;

    # <editor-fold desc="Property: 'md5_string'">
    @property
    def _md5(self):
        if(self._md5_string is None):
            self._md5 = self._get_file_path_md5_string();
        return self._md5_string;


    def read_created_timestamp(self):
        """
        Read the created timestamp of the file.
        :return: a string in the format YYYY-MM-DD HH:MM:SS
        """
        if(self.is_file()):
            return file_created_timestamp_string_from_path(self.absolute_file_path);
        else:
            return None;

    @property
    def _md5_string(self):
        return self.get_info("md5_string");
    @_md5.setter
    def _md5(self, value):
        self._md5_string=value;
    @_md5_string.setter
    def _md5_string(self, value):
        self.set_info('md5_string', value);
    # </editor-fold>

    def is_file(self):
        return os.path.isfile(self.file_path);

    def is_dir(self):
        return os.path.isdir(self.file_path);

    # @property
    # def file_path(self):
    #     return self.get_file_path();
    @property
    def file_name(self):
        return self.get_file_name()
    @property
    def file_name_base(self):
        return self.get_file_name_base()
    @property
    def file_ext(self):
        return self.get_file_extension();

    @property
    def absolute_file_path(self):
        return self.get_absolute_path()


    def get_absolute_path(self, from_root=None):
        """
        Get the absolute path of the file. The optional from_root can be used if the stored file path string is already
        relative to some root
        :param from_root:
        :return:
        """
        if (from_root is None):
            return os.path.abspath(self.file_path)
        else:
            return os.path.abspath(os.path.join(from_root, self.file_path));



    def get_path_with_different_extension(self, new_ext):
        current = self.file_path;
        return os.path.splitext(current)[0] + new_ext;


    def on_path_change(self, new_path=None, old_path=None, **kwargs):
        """
        Anything that should be done when _set_file_path changes the path.
        :param new_path:
        :param old_path:
        :param kwargs:
        :return:
        """
        self._md5 = None;
        return;

    def get_file_name(self):
        filepath = self.file_path
        if (filepath is not None):
            return os.path.basename(self.file_path);

    def get_parent_dir_name(self):
        return os.path.split(self.get_directory_path())[-1]

    def get_file_extension(self):
        filename = self.file_name
        if (filename is not None):
            name_parts = os.path.splitext(self.get_file_name());
            return name_parts[1];

    def get_file_name_base(self):
        filename = self.file_name
        if (filename is not None):
            name_parts = os.path.splitext(self.get_file_name());
            return name_parts[0];

    def get_directory_path(self):
        filepath = self.file_path
        if (filepath is not None):
            if(_myisdir(self.file_path)):
                return self.file_path;
            else:
                return os.path.dirname(self.file_path)+ os.sep;

    def get_directory_name(self):
        return os.path.basename(self.get_directory_path());

    def to_dictionary(self):
        d = super(HasFilePath, self).to_dictionary();
        # d.update(dict(file_path=self.file_path));
        return d;

    def init_from_dictionary(self, d):
        super(HasFilePath, self).init_from_dictionary(d);
        # self._file_path = d['file_path'];

    def show_in_finder(self):
        try:
            if(ptlreg.apy.defines.HAS_FILEUI):
                ptlreg.apy.afileui.Show(self.absolute_file_path);
        except NameError as e:
            print(e)

    # def openFile(self):
    #     fileui.Open(self.file_path);

    def _get_file_path_md5_string(self):
        return hashlib.md5(self.file_path.encode('utf-8')).hexdigest();

    def get_path_with_suffix(self, suffix, extension=None):
        if(extension is None):
            extension = self.get_file_extension();
        return os.path.join(self.get_directory_path(), self.get_file_name_base() + suffix + extension);


