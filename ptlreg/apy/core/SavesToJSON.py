import os

from .filepath import HasFilePath, FilePath
from ptlreg.apy.core import AObject

import ptlreg.apy.utils
# myjsonpickler = jsonpickle.JSONPluginMgr()
import json


def load_from_json(path):
    json_text = open(path).read();
    d = json.loads(json_text);
    cls = AObject.class_from_dictionary(d);
    return cls(path);
    # return AObject.CreateFromDictionary(d);

def load_jsons_from_directory(path):
    if(path is None):
        return None;
    assert(os.path.isdir(path)), "{} is not a directory".format(path);
    return_dict = {};
    for filename in os.listdir(path):
        if(filename.lower().endswith('.json')):
            json_path = os.path.join(path, filename);
            # print(json_path);
            obj = load_from_json(path=json_path);
            if(return_dict.get(obj.aobject_type_name()) is None):
                return_dict[obj.aobject_type_name()]=[obj];
            else:
                return_dict[obj.aobject_type_name()].append(obj);
    return return_dict;


class SavesToJSONMixin(HasFilePath):
    """SavesToJSON -- Mixin to make sure to_dictionary is saved to a JSON, and init_from_dictionary initialized from json.

    Stores the path and filename information about the json file. Saves/loads to/from json.
    """

    # @staticmethod
    # def AObjectType():
    #     return 'SavesToJSON';

    @classmethod
    def _CLASS_JSON_FILE_EXTENSION(cls):
        return '.json';

    def __init__(self, path=None, **kwargs):
        super(SavesToJSONMixin, self).__init__(path=path, **kwargs)

    # <editor-fold desc="Property: 'json_path'">
    @property
    def json_path(self):
        return self._get_json_path();
    def _get_json_path(self):
        return self.file_path;
    # </editor-fold>


    def _get_adjusted_input_file_path(self, input_file_path):
        return super(SavesToJSONMixin, self)._get_adjusted_input_file_path(input_file_path);


    def _init_after_file_path_set(self, **kwargs):
        """
        Here we are going to deal with the case where the json file records a different file_path than the one it is at,
        indicating that the file was not originally saved in its current location.
        :param kwargs:
        :return:
        """
        oldpath = None
        newpath = self.file_path;

        if (os.path.isfile(newpath)):
            self.load_from_json(newpath);
            self.old_path = self.file_path;
        elif(os.path.isdir(newpath)):
            json_file_path = newpath + os.sep + self.default_json_name();
            if (os.path.isfile(json_file_path)):
                self.load_from_json(json_file_path);
                self.old_path = self.file_path;
            newpath = json_file_path;
        else:
            fpath = FilePath(self.file_path);
            if(not fpath.file_ext==self._CLASS_JSON_FILE_EXTENSION()):
                assert False, "Given SavesToJSON path {} for class {} has extension '{}'! should have extension {}\n(in SavesToJSON.py)".format(file_path,
                                                                                                                                                self.__class__.__name__,
                                                                                                                                                fpath.file_ext,
                                                                                                                                                self._CLASS_JSON_FILE_EXTENSION());
        if ((oldpath is not None) and (newpath is not None) and not os.path.samefile(os.path.abspath(oldpath),os.path.abspath(newpath))):
            ptlreg.apy.utils.AINFORM("Detected Path Change in {} from:\nold:{}\nnew:{}".format(self.__class__, oldpath, newpath));
            self.on_path_change(new_path=newpath, old_path=oldpath);
        super(SavesToJSONMixin, self)._init_after_file_path_set(**kwargs);



    def on_path_change(self, new_path=None, old_path = None, **kwargs):
        """
        Anything that should happen when _set_file_path changes the path. For example, if we save an object to a json, move the json, and then load it, then the path will change from where the object was last saved to where the object was last loaded.
        :param new_path:
        :param old_path:
        :param kwargs:
        :return:
        """

        if ((old_path is not None) and (new_path is not None) and (os.path.abspath(old_path) != os.path.abspath(new_path))):
            self.write_to_json(json_path=new_path);
        return super(SavesToJSONMixin, self).on_path_change(new_path=new_path, old_path=old_path, **kwargs);



    def load_from_json(self, json_path=None):
        if (json_path):
            json_text = open(json_path).read();
            # jp = json.loads(json_text);
            # print(json_text);
            d = ptlreg.apy.utils.jsonpickle.decode(json_text);
            self.init_from_dictionary(d);
        # if(json_path):
        #     self._set_file_path(file_path=json_path);
        # if(self.get_file_path()):
        #     json_text=open(self.get_file_path()).read();
        #     d = json.loads(json_text);
        #     self.init_from_dictionary(d);

    def write_to_json(self, json_path=None):
        #with open(jsonpath+self.name+'.json', 'w') as outfile:
        # print('\n')
        # print(self.to_dictionary())
        if(not json_path):
            json_path = self.get_file_path();
        if(json_path):
            with open(json_path, 'w') as outfile:
                dout = ptlreg.apy.utils.jsonpickle.encode(self.to_dictionary());
                outfile.write(dout);
                # json.dump(dout, outfile, sort_keys = True, indent = 4, ensure_ascii=False);
                # print(dout)
                # json.dump(dout, outfile);

        return json_path;

    def save_json(self, json_path=None, **kwargs):
        """This is a wrapper to add other things you might want to do in subclasses"""
        return self.write_to_json(json_path = json_path, **kwargs);

    def save(self, **kwargs):
        if (hasattr(super(SavesToJSONMixin, self), 'save')):
            super(SavesToJSONMixin, self).save(**kwargs);
        self.save_json(**kwargs);

    @classmethod
    def default_json_name(cls):
        return cls.aobject_type_name() + cls._CLASS_JSON_FILE_EXTENSION();



    @classmethod
    def from_json(cls, json_path=None):
        rval = cls();
        rval.load_from_json(json_path=json_path);
        return rval;

class SavesToJSON(SavesToJSONMixin, AObject):
    def set_file_path(self, file_path=None, **kwargs):
        super(SavesToJSON, self).set_file_path(file_path=file_path, **kwargs)
