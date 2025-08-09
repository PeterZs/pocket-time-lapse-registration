from ptlreg.apydn.HasFilePathLabel import *
import errno



class FPath(HasFilePathLabel, HasDataLabels):
    def __init__(self, path=None, *args, **kwargs):
        super(FPath, self).__init__(path=path, *args, **kwargs);
    def __str__(self):
        return self.file_path;
    def __repr__(self):
        return '[FPath]:{}'.format(self.file_path);

    @classmethod
    def From(cls, path):
        if(isinstance(path, cls)):
            return path.clone();
        else:
            return cls(path);

    @classmethod
    def from_arg(cls, path):
        if (isinstance(path, cls)):
            return path.clone();
        else:
            return cls(path);

    @staticmethod
    def get_dir_from_path(pth):
        return (os.path.split(pth)[0] + os.sep);

    @staticmethod
    def make_sure_dir_path_exists(path):
        """

        :param path:
        :return:
        """
        # TODO Add support for FPATH here
        pparts = os.path.split(path);  # Does this return bytes?
        destfolder = pparts[0] + os.sep;
        try:
            os.makedirs(destfolder)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    @staticmethod
    def make_sure_path_exists(path):
        """

        :param path:
        :return:
        """
        pathstring = path;
        if(isinstance(path, HasFilePathLabel)):
            pathstring = path.get_absolute_path();
        try:
            os.makedirs(pathstring)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def make_sure_dir_exists(self):
        mydir = self.get_directory_path();
        self.__class__.make_sure_dir_path_exists(mydir);

    def to_dictionary(self):
        d = super(HasFilePathLabel, self).to_dictionary();
        # d.update(dict(file_path=self.file_path));
        return d;



    @classmethod
    def path_exists(cls, filepath):
        fp = cls.From(filepath);
        return fp.file_exists();

    @property
    def looks_like_dir(self):
        """
        whether the path looks like a directory. so 'a/b/c/' and 'a/b/c' both return true, but 'a/b/c.ext' returns false.
        :return:
        """
        if(self.file_path is None):
            return self.file_path;
        return (self.file_path == os.path.splitext(self.file_path)[0]);




    def get_with_file_name_suffix(self, suffix):
        return FPath(os.path.join(self.get_directory_path(), self.get_file_name_base() + suffix + self.get_file_extension()))


# class FPath(FilePathLabel):
#     def __init__(self, path=None, *args, **kwargs):
#         super(FPath, self).__init__(path=path, *args, **kwargs);