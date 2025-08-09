from ptlreg.apy.core.tests.test_helper import *
import unittest

testdir = GetTestDir(__file__);

from functools import total_ordering
@total_ordering
class Ev(AObject):
    """
    """
    @staticmethod
    def AObjectType():
        return 'Ev';
    def __init__(self, event_time=None, **kwargs):
        self._event_time = None;
        super(Ev, self).__init__(**kwargs)
        self.event_time = event_time;
        self.update_info(kwargs);
    def __repr__(self):
        return "Ev EventTime: {}".format(self.event_time);
    def __eq__(self, other):
        if(isinstance(other, Ev)):
            return (self._get_as_dictionary_string().lower() == other._get_as_dictionary_string().lower());
        else:
            return (self.event_time == other);
    def __lt__(self, other):
        return self.event_time<other.event_time;
    # <editor-fold desc="Property: 'event_time'">
    @property
    def event_time(self):
        return self._getEventTime();
    def _getEventTime(self):
        return self._event_time;
    @event_time.setter
    def event_time(self, value):
        self._setEventTime(value);
    def _setEventTime(self, value):
        self._event_time = value;
    # </editor-fold>


class ListTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_sure_dir_exists(testdir);

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(testdir)
        return;

    def testAObjectList(self):
        l = AObjectList();

        regular_list = [];
        for i in range(200):
            regular_list.append(np.random.randint(200));
            l.append(Ev(regular_list[-1]));

        self.assertEqual(l.list_attribute('event_time'), regular_list);

        for i,a, in enumerate(l):
            self.assertEqual(regular_list[i],a);

        regular_list.sort();
        l.sort()

        for i, a, in enumerate(l):
            self.assertEqual(regular_list[i], a);

    def testAObjectListSaveLoad(self):
        TestAObjectLists(testdir);




def main():
    unittest.main();

if __name__ == '__main__':
    main()
