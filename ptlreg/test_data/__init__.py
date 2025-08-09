import os;
def _get_test_data_dir():
    return os.path.abspath(os.path.dirname(__file__));


def get_structured_test_data_dir():
    return os.path.join(_get_test_data_dir(), "structured");

def get_structured_test_data_primary_dir():
    return os.path.join(get_structured_test_data_dir(), "primary");

def get_structured_test_data_secondary_dir():
    return os.path.join(get_structured_test_data_dir(), "secondary");

def get_unstructured_test_data_dir():
    return os.path.join(_get_test_data_dir(), "unstructured");




