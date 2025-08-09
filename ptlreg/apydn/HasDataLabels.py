import numpy as np
import pandas as pd
import pandera.pandas as pa
import os, warnings


class LabelValidationError(Exception):
    def __init__(self, labelKey, value, message=None):
        self.key = labelKey;
        self.value = value;
        self.message=message;
        super(LabelValidationError, self).__init__(self.message)




class HasDataLabels(object):
    """
    Has data labels mixin class.
    """

    DATALABEL_SCHEMA = pa.DataFrameSchema();
    DATALABELS_DICT_KEY = "data_labels";
        # "__data_labels__"

    @classmethod
    def VALIDATE_TIMESTAMP(cls, value):
        # warnings.warn("VALIDATE_TIMESTAMP not implemented", UserWarning)
        return value;

    def __init__(self, *args, data_labels=None, **kwargs):
        self._set_data_labels(data_labels);
        super(HasDataLabels, self).__init__(*args, **kwargs)
        if(self.data_labels is None):
            self._set_data_labels(self.__class__._new_series());


    @classmethod
    def _new_series(cls, *args, **kwargs):
        return pd.Series(*args, **kwargs);


    def get_serializable_value_dict_for_label_keys(self, label_keys=None):
        if(label_keys is None):
            label_keys = self.data_labels.index.tolist();
        rdict = {};
        for k in label_keys:
            value = self.get_label_value(k);
            if(isinstance(value, np.ndarray)):
                rdict[k] = value.tolist();
            else:
                rdict[k] = value;
        return rdict;


    @classmethod
    def validate_data_labels_for_instance(cls, data_labels):
        if(isinstance(data_labels, HasDataLabels)):
            return cls.DATALABEL_SCHEMA.validate(data_labels.data_labels);
        return cls.DATALABEL_SCHEMA.validate(data_labels);

    def validate_data_labels(self):
        return self.__class__.validate_data_labels_for_instance(self.data_labels);

    def get_data_labels_dataframe(self, index=None):
        if(index is None):
            return pd.DataFrame([self.data_labels]);
        else:
            return pd.DataFrame([self.data_labels], index=index);

    def _set_data_labels(self, data_labels: pd.Series=None):
        if(isinstance(data_labels, pd.Series)):
            self._data_labels = data_labels;
        else:
            self._data_labels = self.__class__._new_series(data_labels);

    @property
    def data_labels(self)-> pd.Series|None:
        return self._data_labels;


    @classmethod
    def from_series(cls, series):
        return cls(data_labels=series);

    # def get_label(self, key):
    #     return self.data_labels[key];

    def set_label(self, key, value):
        self.data_labels[key]=value;

    def has_label(self, key):
        return key in self.data_labels;

    # def has_label(self, key):
    #     return key in self.data_labels._main_index_map;

    def get_label(self, key):
        """
        Wrapper for data_labels that returns None if:
         - the label is not set
         - the label is NAN
         - the label is None
        :param key:
        :return:
        """
        if((key in self.data_labels) and (self.data_labels is not np.NaN)):
            return self._get_label(key);
        else:
            return None;

    def get_label_value(self, key):
        """
        Returns the value of the label for the given key.
        :param key:
        :return:
        """
        # warnings.warn("get_label_value is deprecated. Used get_label", DeprecationWarning);
        return self.get_label(key);

    def set_label_value(self, key, value):
        """
        Sets the value of the label for the given key.
        :param key:
        :param value:
        :return:
        """
        # This had another level of indirection before, but now we just set the value directly
        self.data_labels[key] = value;
        # self.data_labels.get(key).setValue(value);



    def data_labels_for_keys(self, label_keys=None):
        """
        Returns a pandas Series with the data labels for the given keys.
        If label_keys is None, returns all data labels.
        :param label_keys:
        :return:
        """
        if(label_keys is None):
            return self.data_labels;
        else:
            return self.data_labels[label_keys];

    def _get_label(self, key):
        return self.data_labels[key]

    def has_tag(self, key):
        return (self.has_label(key) and (self.get_label(key)));

    def add_label(self, key, value=True):
        if(self.has_label(key)):
            raise LabelValidationError(labelKey=key, value=value, message="Tried to add label that already exists");
        else:
            self.set_label(key, value);
        # self.data_labels.append(labelInstance);

    # def add_label_instance(self, labelInstance):
    #     self.data_labels.append(labelInstance);

    def add_tag_label(self, tag):
        if (self.has_label(tag)):
            raise LabelValidationError(labelKey=tag, value=True, message="Tried to add label that already exists");
        else:
            self.set_label(tag, True);

    def set_tag_label(self, tag, value=True):
        self.set_label(tag, value);

    def add_string_label(self, key, value):
        if value is not type(str):
            value = str(value)
        self.add_label(key, value);
        # self.data_labels.append(StringLabel(key, value, visible, editable));

    def add_timestamp_label(self, key, value, visible=False, editable=False):
        timestamp = self.__class__.VALIDATE_TIMESTAMP(value);
        self.set_label(key, timestamp);


    def to_dictionary(self):
        d = {};
        if (hasattr(super(HasDataLabels, self), 'to_dictionary')):
            d = super(HasDataLabels, self).to_dictionary();
        d[self.__class__.DATALABELS_DICT_KEY]=self.data_labels.to_dict();
        return d;




    def init_from_dictionary(self, d):
        if (hasattr(super(HasDataLabels, self), 'init_from_dictionary')):
            super(HasDataLabels, self).init_from_dictionary(d);
        self._set_data_labels(self.__class__._new_series(d[self.__class__.DATALABELS_DICT_KEY]));


    # </editor-fold>