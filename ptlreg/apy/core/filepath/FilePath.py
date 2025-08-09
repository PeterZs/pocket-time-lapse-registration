from ..aobject import *
from .HasFilePath import *

class FilePath(HasFilePath, AObject):
    ALLOW_DIRECTORIES = True
    def __init__(self, path=None, **kwargs):
        super(FilePath, self).__init__(path=path, **kwargs);
    def __str__(self):
        return self.file_path;
    def __repr__(self):
        return '[FilePath]:{}'.format(self.file_path);

    @classmethod
    def from_arg(cls, path):
        if(isinstance(path, cls)):
            return path.clone();
        else:
            return cls(path);

    @classmethod
    def From(cls, path):
        if (isinstance(path, cls)):
            return path.clone();
        else:
            return cls(path);


    def exists(self):
        return os.path.exists(self.file_path)

    @classmethod
    def file_exists(cls, filepath):
        fp = FilePath.from_arg(filepath);
        return fp.exists();

    @property
    def looks_like_dir(self):
        """
        whether the path looks like a directory. so 'a/b/c/' and 'a/b/c' both return true, but 'a/b/c.ext' returns false.
        :return:
        """
        if(self.file_path is None):
            return self.file_path;
        return (self.file_path == os.path.splitext(self.file_path)[0]);


    def relative(self, from_path=None):
        if(from_path is None):
            return os.path.relpath(self.file_path);
        else:
            return os.path.relpath(self.file_path, from_path);

    def get_with_file_name_suffix(self, suffix):
        return FilePath(os.path.join(self.get_directory_path(), self.get_file_name_base() + suffix + self.get_file_extension()))
