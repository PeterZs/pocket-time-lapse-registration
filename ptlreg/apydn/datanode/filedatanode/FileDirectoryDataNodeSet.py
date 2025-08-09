from ptlreg.apydn.datanode.filedatanode.FileDataNode import HasFileDataNode_Mixin, FileDataNode
from ptlreg.apydn.datanode.DataNodeSet import DataNodeSet
from .FileDataNodeSet import HasFileDataNodeSet_Mixin
from ... import FPath


def _endswithany(fstring, extensions):
    for e in extensions:
        if(fstring.lower().endswith(e)):
            return True;
    return False;


class FileDirectoryDataNodeSet(HasFileDataNode_Mixin, HasFileDataNodeSet_Mixin, DataNodeSet):
    """
    A DirectoryDataNodeSet is a set of FileDataNodes that represent files in a directory.
    It is a subclass of HasFileDataNodeSet_Mixin and DirectoryDataNodeSet.
    """

    DATANODE_CLASS = FileDataNode;  # individual elements are this class

    def __init__(self, path=None, root_path = None, *args, **kwargs):
        """
        Initialize FileDataNode. This is a node that represents a file with a path.
        :param path: The path
        :param root_path: a root path to store this path relative to
        :param kwargs:
        """
        super(FileDirectoryDataNodeSet, self).__init__(path=path, root_path=root_path, *args, **kwargs);


    @classmethod
    def from_directory(cls, directory, root_path=None, recursive=False, extension_list=None, criteriaFunc=None,
                       include_hidden_files=False):
        """

        :param directory: absolute path to directory
        :param root_path: root path to use
        :param recursive:
        :param extension_list:
        :param criteriaFunc:
        :param include_hidden_files:
        :return:
        """
        if (root_path is None):
            root_path = directory;
        new_nodeset = cls(path=directory, root_path=root_path);
        if (not recursive):
            for filename in os.listdir(new_nodeset.get_absolute_path()):
                if ((extension_list is None) or _endswithany(filename, extension_list)):
                    fpath = FPath(os.path.join(os.path.abspath(directory), filename));
                    include_fpath = True;
                    if (not include_hidden_files and fpath.file_name[0] == "."):
                        include_fpath = False;
                    if (criteriaFunc):
                        include_fpath = include_fpath and criteriaFunc(fpath);
                    if (include_fpath):
                        # print("root path should be {}".format(new_nodeset.absolute_file_path));
                        newNode = cls.DATANODE_CLASS(path=fpath.relative_path(to_path=new_nodeset.root_path),
                                                     root_path=new_nodeset.absolute_file_path);
                        # print(newNode.data_labels)
                        new_nodeset.add_node(newNode);
            return new_nodeset;
        else:
            raise NotImplementedError;
        #     for root, dirs, files in os.walk(directory):
        #         for f in files:
        #             if ((extension_list is None) or _endswithany(f, extension_list)):
        #                 fpth = FPath(os.path.join(os.path.abspath(directory), f));
        #                 FilePath(os.path.join(root, f));
        #                 if (criteriaFunc):
        #                     if (criteriaFunc(fpth)):
        #                         new_list.append(fpth);
        #                 else:
        #                     new_list.append(fpth);
        # return new_list;
