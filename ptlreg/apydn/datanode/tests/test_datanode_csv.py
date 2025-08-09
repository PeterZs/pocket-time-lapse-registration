import unittest
import os
import pandas as pd
from ptlreg.apydn.datanode import DataNode

class TestDataNode(unittest.TestCase):
    def setUp(self):
        self.test_csv = 'test_datanode.csv'
        self.labels = {
            'label1': 'value1',
            'label2': 42,
            'label3': 3.14
        }
        self.node = DataNode()
        for k, v in self.labels.items():
            self.node.set_label(k, v)

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def test_save_and_load(self):
        # Save to CSV
        self.node.save_data_labels_to_csv(self.test_csv)
        # Load from CSV
        loaded_node = DataNode()
        loaded_node.load_data_labels_from_csv(self.test_csv)
        # Check labels
        for k, v in self.labels.items():
            self.assertEqual(loaded_node.get_label(k), v)

if __name__ == '__main__':
    unittest.main()