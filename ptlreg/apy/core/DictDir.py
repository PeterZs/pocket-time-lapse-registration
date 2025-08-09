from future.utils import iteritems

from .SavesToJSON import SavesToJSON




class DictDir(SavesToJSON):
    """
    Treat this as a dictionary that you can save and load easily..
    It also prints more nicely and you can use addItem and getItem with a list of keys.
    """
    def __init__(self, base_dictionary = None, **kwargs):
        super(DictDir, self).__init__(**kwargs);
        if(base_dictionary is not None):
            self.dictionary = base_dictionary;
        else:
            self.dictionary = {};

    def __str__(self):
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

        return formatcontainer(self.dictionary);


    # <editor-fold desc="Property: 'dictionary'">
    @property
    def dictionary(self):
        return self._dictionary;
    @dictionary.setter
    def dictionary(self, value):
        self._dictionary = value;

    @property
    def _dictionary(self):
        return self.get_info('dictionary');
    @_dictionary.setter
    def _dictionary(self, value):
        self.set_info('dictionary', value);
    # </editor-fold>


    def add_item(self, item_detail_list, item_value):
        ldict = self.dictionary;
        if (not isinstance(item_detail_list, (list, tuple))):
            searchlist = [item_detail_list]
        else:
            searchlist = item_detail_list;
        for i, od in enumerate(searchlist):
            if (ldict.get(od) is None):
                if (i == (len(searchlist) - 1)):
                    ldict[od] = item_value;
                else:
                    ldict[od] = {};
            ldict = ldict[od];

    def get_item(self, item_detail_list):
        ldict = self.dictionary;
        searchlist = item_detail_list;
        if (not isinstance(item_detail_list, (list, tuple))):
            searchlist = [item_detail_list]
        for i, od in enumerate(searchlist):
            if(not isinstance(ldict, dict)):
                return None;
            if (ldict.get(od) is None):
                return None;
            ldict = ldict[od];
        return ldict;

    def to_dictionary(self):
        d = super(DictDir, self).to_dictionary();
        #d['']=self.
        return d;


    def init_from_dictionary(self, d):
        super(DictDir, self).init_from_dictionary(d);
        #self. = d[''];

    def __getitem__(self, key):
        return self.dictionary.__getitem__(key);

    def __setitem__(self, key, value):
        return self.dictionary.__setitem__(key, value);

    def __delitem__(self, key):
        return self.dictionary.__delitem__(key);

    def __missing__(self, key):
        return self.dictionary.__missing__(key);

    def __iter__(self):
        return self.dictionary.__iter__();

    def __contains__(self):
        return self.dictionary.__contains__();

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.dictionary== other.dictionary;
        return self.dictionary== other

    def get(self, *args, **kwargs):
        return self.dictionary.get(*args, **kwargs);

    def iteritems(self):
        return iteritems(self.dictionary);
        # return self.dictionary.iteritems();


    def __call__(self, *detail_list):
        # if (not isinstance(value, (list, tuple))):
        #     op_domains = [op_domains];
        return self.get_item(detail_list);
