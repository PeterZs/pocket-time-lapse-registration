import uuid

def GetUUID():
    return uuid.uuid1();

class UniqueID(object):

    def __init__(self,**kwargs):
        self._uid = GetUUID();
        super(UniqueID, self).__init__(**kwargs);

    @property
    def uid(self):
        return str(self._uid);

    def to_dictionary(self):
        d = super(UniqueID, self).to_dictionary();
        d['_uid']=self._uid;
        return d;

    def init_from_dictionary(self, d):
        super(UniqueID, self).init_from_dictionary(d);
        self._uid = d['_uid'];