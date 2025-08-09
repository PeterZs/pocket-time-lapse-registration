from ptlreg.apy.core import SavesFeatures
from ptlreg.apy.core import HasFilePath


class SignalMixin(HasFilePath, SavesFeatures):
    """
    This is the parent class that defines abstractions for dealing with a Signal.
    """
    def __init__(self, name=None, *args, **kwargs):
        # Don't know what _samples will be yet, but it should be something...
        self._samples = None;
        if(not hasattr(self, "name")):
            if(name is not None):
                self.name = name;
            else:
                self.name = ""

        super(SignalMixin, self).__init__(*args, **kwargs);

    # Must define _samples
    # <editor-fold desc="Property: 'samples'">
    @property
    def samples(self):
        return self._get_samples();
    def _get_samples(self):
        return self._samples;
    @samples.setter
    def samples(self, value):
        self._set_samples(value);
    def _set_samples(self, value):
        self._samples = value;
    # </editor-fold>