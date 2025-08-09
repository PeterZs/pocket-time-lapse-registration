


class HasTags(object):

    @classmethod
    def _tags_arg_to_tags_dict(cls, tags=None):
        if(tags is None):
            return {};
        if(isinstance(tags, dict)):
            return tags;
        elif(isinstance(tags, (list, tuple))):
            d = {};
            for t in tags:
                d[t]=True;
            return d
        else:
            return {tags:True};





    def __init__(self, tags=None, **kwargs):
        """

        :param tags:
        :param kwargs:
        """
        self._tags = {};
        super(HasTags, self).__init__(**kwargs);
        if(tags is not None):
            assert((len(self.get_tags()) == 0) or (tags == self.get_tags())), "tried to init tags: {}\nFor HasTags {}\nBut it already had tags: {}".format(tags, self, self.get_tags());
        self.update_tags(tags);


    @property
    def tags(self):
        return self._tags;

    # @tags.setter
    # def tags(self, value):
    #     self._tags = value;


    # def set_tags(self, tags):
    #     for nt in tags:
    #         self.tags[nt]=tags[nt];

    def get_tag(self, tag):
        return self._tags.get(tag);
    def set_tag(self, tag, tag_value=True):
        self._tags[tag]=tag_value;
        if(hasattr(self, tag)):
            assert(False), "{} cannot be set as a tag, because it is is already an attribute of {}".format(tag, self);

    def get_tags(self):
        return self._tags;

    def clear_tags(self, tags=None):
        if(tags is None):
            self._tags = {};
        else:
            for r in tags:
                del self._tags[t];

    def set_tags(self, tags=None):
        """
        does a dict.update
        :param tags:
        :return:
        """
        if(tags is not None):
            self._tags.update(tags);

    def update_tags(self, tags=None):
        utags = self._tags_arg_to_tags_dict(tags);
        self._tags.update(utags);

    def to_dictionary(self):
        d = super(HasTags, self).to_dictionary();
        d['tags']=self._tags;
        return d;

    def init_from_dictionary(self, d):
        super(HasTags, self).init_from_dictionary(d);
        self._tags = d['tags'];



    # </editor-fold>