"""
XML Book Tools Module
=====================

This library implements a collection of helper functions used by the book.
"""

# Author: Kacper Sokol <kacper@xmlx.io>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: MIT

import os
import sys

__all__ = ['initialise_colab']

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

try:
    import fatf
    FATF_INSTALLED = True
except ImportError:
    FATF_INSTALLED = False

DEPS_INSTALLED = False


def initialise_colab():
    """
    Installs FAT Forensics with all its auxiliary dependencies when in Colab.
    """
    global FATF_INSTALLED
    global DEPS_INSTALLED

    if IN_COLAB:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.abspath(os.path.join(file_dir, '../'))
        requirements_file = os.path.join(root_dir, 'requirements-code.txt')

        print('Installing dependencies.')
        stream = os.popen(
            '{} -m pip install -r {}'.format(sys.executable, requirements_file))
        output = stream.read()
        print(output)

        FATF_INSTALLED = True
        DEPS_INSTALLED = True
    else:
        print('Not in Colab; nothing to do.')
