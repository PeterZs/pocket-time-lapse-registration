import unittest
import ptlreg.apy.core.tests.test_saving as test_saving
import ptlreg.apy.core.tests.test_loading as test_loading

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_saving))
suite.addTests(loader.loadTestsFromModule(test_loading))

def main():
    print("CALLING MAIN IN __MAIN__!");
    return test_saving.main();

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)
