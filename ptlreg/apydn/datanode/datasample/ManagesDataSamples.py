from ptlreg.apydn.datanode.datasample import DataSampleSet


class ManagesDataSamples(object):
    """
    Mixin to make an object manage data samples. Assumes the class will be mixed with an AObject, e.g., SavesDirectories.
    """
    SAMPLES_KEY = "samples"
    SAMPLE_SET_TYPE = DataSampleSet;

    @classmethod
    def create_sample_set(cls, *args, **kwargs):
        return cls.SAMPLE_SET_TYPE(*args, **kwargs);

    @classmethod
    def create_sample(cls, *args, **kwargs):
        return cls.SAMPLE_SET_TYPE.new_element(*args, **kwargs);

    def __init__(self, *args, **kwargs):
        self.samples = None;
        super(ManagesDataSamples, self).__init__(*args, **kwargs)
        if (self.samples is None):
            self.samples = self.__class__.create_sample_set();

    # <editor-fold desc="Property: 'samples'">
    @property
    def samples(self):
        """
        one level of indirection to allow redirecting this in subclasses
        :return:
        """
        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = value;

    # </editor-fold>

    def to_dictionary(self):
        d = super(ManagesDataSamples, self).to_dictionary();
        d[self.__class__.SAMPLES_KEY] = self.samples.to_dictionary();
        return d;

    def init_from_dictionary(self, d):
        super(ManagesDataSamples, self).init_from_dictionary(d);
        samples = self.__class__.SAMPLE_SET_TYPE.from_dictionary(d[self.__class__.SAMPLES_KEY]);
        self.samples = samples;


