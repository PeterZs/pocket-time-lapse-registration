# python
import unittest
import pandas as pd
from pandera.errors import SchemaError

from ptlreg.apydn.datanode import *

test_data = {"node_id": "test_id", "name": "Test Node"}

class TestDataNode(unittest.TestCase):
    def setUp(self):
        # Example data for testing
        self.data_labels = pd.Series(test_data)
        self.node = DataNode(data_labels=self.data_labels)

    def test_initialization(self):
        # Test if the DataNode is initialized correctly
        self.assertEqual(self.node["node_id"], "test_id")
        self.assertEqual(self.node["name"], "Test Node")

    def test_create_with_id(self):
        # Test creating a DataNode with a specific ID
        new_node = DataNode.create_with_id("new_id", data_labels={"name": "New Node"})
        self.assertEqual(new_node["node_id"], "new_id")
        self.assertEqual(new_node["name"], "New Node")

    def test_validation(self):
        # Test validation of data labels
        valid_labels = pd.Series({"node_id": "valid_id", "name": "Valid Node"})
        try:
            validated = DataNode.validate_data_labels_for_instance(valid_labels);
            self.assertTrue(True, "valid data labels");
        except SchemaError:
            self.assertTrue(False, SchemaError);


    def test_serialization_to_csv(self):
        # Test saving data labels to a CSV file
        path = "test_data_labels.csv"
        self.node.save_data_labels_to_csv(path)
        loaded_node = DataNode.from_csv(path)
        self.assertEqual(loaded_node["node_id"], "test_id")
        self.assertEqual(loaded_node["name"], "Test Node")

    def test_data_access(self):
        # Test accessing and modifying data labels
        self.assertEqual(self.node["name"], "Test Node")
        self.node["name"] = "Updated Node"
        self.assertEqual(self.node["name"], "Updated Node")

    def tearDown(self):
        # Clean up any created files
        import os
        if os.path.exists("test_data_labels.csv"):
            os.remove("test_data_labels.csv")

if __name__ == "__main__":
    unittest.main()