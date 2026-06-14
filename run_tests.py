"""Wrapper to run test_notebook.py with Python 3.14 compatibility patches."""
import collections.abc
import collections
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable

import sys
sys.path.insert(0, 'src')

import runpy
runpy.run_path('test_notebook.py', run_name='__main__')
