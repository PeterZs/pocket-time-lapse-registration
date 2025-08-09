# import jsonpickle


# SERIALIZABLE_VALUE_STRING = 'SerializableType'

class Serializable(object):
    AINFO_DICT_KEY = '_ainfo';

    def __init__(self, **kwargs):
        """
        _ainfo is part of serialization
        _temp_info is not part of the serialization
        :param kwargs:
        """
        assert(not any(kwargs)), "unhandled keyword arguments {} for init on class {}".format(kwargs, self.__class__.__name__);
        # super(Serializable, self).__init__(**kwargs)
        self._init_ainfo();
        self._temp_info = {};

    def _init_ainfo(self):
        self._ainfo = {};

    def __str__(self):
        return self._get_as_dictionary_string();

    def set_info(self, label, value):
        """

        :param label: label for the info being set
        :param value: value info is being set to
        :return:
        """
        # assert(label != SERIALIZABLE_AINFO_KEY_ID)
        self._ainfo[label]=value;

    def update_info(self, d):
        self._ainfo.update(d);

    def has_info(self, label):
        """

        :param label: label being checked
        :return:
        """
        return (label in self._ainfo);

    def get_info(self, label):
        """

        :param label: label of info being retrieved
        :return: value of info with that label
        """
        return self._ainfo.get(label);

    def _get_ainfo_dict(self):
        return dict(self._ainfo);

    def _serialize_info(self):
        """

        :return: the _ainfo dict,
        """
        return self._get_ainfo_dict()
        # rdict = dict(self._ainfo);
        # has_serializable = False;
        # for k in rdict:
        #     if(isinstance(k, Serializable)):
        #         rdict[k] = rdict[k].to_dictionary();
        #         has_serializable = True;
        #         rdict[k][SERIALIZABLE_AINFO_KEY_ID] = SERIALIZABLE_AINFO_VALUE_ID;
        #
        # return rdict;
        # return dict(self._ainfo);
        # return jsonpickle.encode(self._ainfo);

    @staticmethod
    def _decode_info(d):
        return d;
        # rdict = dict(d);
        # # if (rdict[k].get(SERIALIZABLE_AINFO_KEY_ID) == SERIALIZABLE_AINFO_VALUE_ID):
        # for k in rdict:
        #     if(isinstance(rdict[k], dict)):
        #         if(rdict[k].get(SERIALIZABLE_AINFO_KEY_ID)==SERIALIZABLE_AINFO_VALUE_ID):
        #             rdict[k] = Serializable.CreateFromDictionary(rdict[k]);
        # return rdict;



    # def ToJSONPickleString(self):
    #     return jsonpickle.encode(self);

    # @staticmethod
    # def FromJSONPickleString(serialized):
    #     return jsonpickle.decode(serialized);


    def to_dictionary(self):
        """
        :return:
        """
        if (hasattr(super(Serializable, self), 'to_dictionary')):
            d = super(Serializable, self).to_dictionary();
        else:
            d = {};
        d.update({self.__class__.AINFO_DICT_KEY: self._serialize_info()});
        return d;


    def init_from_dictionary(self, d):
        if(hasattr(super(Serializable, self), 'init_from_dictionary')):
            super(Serializable, self).init_from_dictionary(d);
        # info_dict = self.__class__._decode_info(d.get('_ainfo'));
        info_dict = self.__class__._decode_info(d.get(self.__class__.AINFO_DICT_KEY, None));
        self.update_info_from_info_dict(info_dict);
        # self._ainfo.update();


    def update_info_from_info_dict(self, info_dict):
        """
        Updates the info of this object from a dictionary.
        :param info_dict: dictionary with info to update
        :return:
        """
        if(info_dict is not None):
            self._ainfo.update(info_dict);


    def print_as_dictionary(self):
        print(self._get_as_dictionary_string());

    def _get_as_dictionary_string(self):
        d = self.to_dictionary();

        def formatcontainer(d, tab=0):
            s = ['{\n']

            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (dict, list)):
                        v = formatcontainer(v, tab + 1)
                    else:
                        v = repr(v)

                    s.append('%s%r: %s,\n' % ('  ' * tab, k, v))
                s.append('%s}' % ('  ' * tab))
            elif(isinstance(d, list)):
                for v in d:
                    if(isinstance(v, (dict, list))):
                        v = formatcontainer(v, tab+1);
                    s.append('%s %s,\n' % ('  ' * tab, v))
                s.append('%s}' % ('  ' * tab))
            return ''.join(s)

        return formatcontainer(d);


    # def ToJSONPickleString(self):
    #     return jsonpickle.encode(self);
    #
    # @staticmethod
    # def FromJSONPickleString(serialized):
    #     return jsonpickle.decode(serialized);