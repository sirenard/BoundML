import unittest

from utils import setup_fake_environment

setup_fake_environment()

raise unittest.SkipTest("Example script is not executed as part of the unit test suite.")
