import unittest
import os
from ptlreg.apy.core.filepath import HasFilePath, FilePath

class TestFilePath(unittest.TestCase):
    def setUp(self):
        self.test_file_path_string = "test_file.txt"
        self.test_root_path_string = os.path.join("", "testfilepath_temp_dir") + os.sep
        self.test_file_path = FilePath.From(os.path.join(self.test_root_path_string, self.test_file_path_string))

    def test_get_absolute_path(self):
        abs_path = self.test_file_path.get_absolute_path()
        self.assertEqual(abs_path, os.path.abspath(self.test_file_path.file_path))

    def test_file_exists(self):
        self.assertTrue(not self.test_file_path.exists())
        os.makedirs(self.test_root_path_string, exist_ok=True)
        abs_path = os.path.join(self.test_root_path_string, self.test_file_path_string);
        with open(abs_path, "w") as f:
            f.write("Test content")
        self.assertTrue(FilePath.From(abs_path).exists())
        os.remove(abs_path)

    # def test_relative_path(self):
    #     rel_path = self.test_file_path.relative_path(to_path="/")
    #     self.assertEqual(rel_path, os.path.relpath(os.path.join(self.test_root_path_string, self.test_file_path_string), "/"))

    def tearDown(self):
        if os.path.exists(self.test_root_path_string):
            os.rmdir(self.test_root_path_string)


if __name__ == "__main__":
    unittest.main()