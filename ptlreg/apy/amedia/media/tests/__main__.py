import unittest
import .test_image

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(apymedia.media.tests.test_image));

def main():
    print("CALLING MAIN IN __MAIN__!");
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)

if __name__ == '__main__':
    main()
