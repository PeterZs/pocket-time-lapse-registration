


class Sequencable(object):
    """
    Mixin for thinks you want to sequence in AObjectList
    """

    # <editor-fold desc="Property: '_prev'">
    @property
    def _prev(self):
        return self._temp_info.get('_prev');
    @_prev.setter
    def _prev(self, value):
        self._temp_info['_prev']=value;
    # </editor-fold>

    # <editor-fold desc="Property: '_next'">
    @property
    def _next(self):
        return self._temp_info.get('_next');
    @_next.setter
    def _next(self, value):
        self._temp_info['_next']=value;
    # </editor-fold>

    # <editor-fold desc="Property: '_list'">
    @property
    def _list(self):
        return self._temp_info.get('_list');
    @_list.setter
    def _list(self, value):
        self._temp_info['_list']=value;
    # </editor-fold>

    # <editor-fold desc="Property: '_id'">
    @property
    def _id(self):
        return self._temp_info.get('_id');
    @_id.setter
    def _id(self, value):
        self._temp_info['_id']=value;
    # </editor-fold>
