#!/usr/bin/env python3

import sys
import os

extensions = []

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

project = 'Test'
copyright = '2020, Florian Flachsenberg, Matthias Rarey'
author = 'Florian Flachsenberg, Matthias Rarey'

version = '1.0'
release = '1.0'

language = None

exclude_patterns = ['_build']

pygments_style = 'sphinx'

todo_include_todos = False

html_theme = 'alabaster'

html_static_path = ['_static']

html_extra_path = ['../html']

htmlhelp_basename = 'Testdoc'

latex_elements = {
}

latex_documents = [
    (master_doc, 'Test.tex', 'Test Documentation',
     'Florian Flachsenberg, Matthias Rarey', 'manual'),
]

man_pages = [
    (master_doc, 'Test', 'Test Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'Test', 'Test Documentation',
     author, 'Test', 'One line description of project.',
     'Miscellaneous'),
]


import subprocess
import shutil
subprocess.call('doxygen documentation/doxygen.cfg', shell=True, cwd='../../')
subprocess.call('make', shell=True, cwd='../latex')
shutil.copy('../latex/refman.pdf', '../html/')
