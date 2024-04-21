""" Provides methods to access various resource files for the application."""

import os.path

BASE_PATH = os.path.dirname(__file__)


def find(rel_path):
    """
    Looks up the resource file based on the relative path to this module.

    :param      relpath | str

    :return     str
    """
    return os.path.join(BASE_PATH, rel_path)