from ptlreg.apy.core import datetime_from_formatted_timestamp_string
from ptlreg.apydn import DataNodeConstants


class HasImageSampleLabelsMixin:
    """
    Mixin class for handling image sample labels.
    """

    def set_timestamp(self, datetime):
        if(self.get_label(DataNodeConstants.CREATED_TIMESTAMP_KEY) is None):
            self.add_timestamp_label(key=DataNodeConstants.CREATED_TIMESTAMP_KEY, value=datetime);
        else:
            self.set_label_value(DataNodeConstants.CREATED_TIMESTAMP_KEY, datetime);

    @property
    def timestamp(self):
        return self.get_label_value(DataNodeConstants.CREATED_TIMESTAMP_KEY);

    @property
    def timestamp_datetime(self):
        return datetime_from_formatted_timestamp_string(self.timestamp)

    @property
    def gps(self):
        return self.get_label_value(ImageSampleConstants.GPS_KEY);

    @gps.setter
    def gps(self, value):
        if (self.labels.get(ImageSampleConstants.GPS_KEY) is None):
            self.addVec2Label(key=ImageSampleConstants.GPS_KEY, value=value);
        else:
            self.set_label_value(ImageSampleConstants.GPS_KEY, value);