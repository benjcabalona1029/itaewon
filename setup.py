import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.2'
PACKAGE_NAME = 'itaewon'
AUTHOR = 'Benjamin Cabalona Jr'
AUTHOR_EMAIL = 'benjcabalonajr@gmail.com'
URL = 'https://github.com/benjcabalona1029/itaewon'

LICENSE = 'MIT License'
DESCRIPTION = 'A personal python library to speed up my workflow'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'sklearn',
      'rpy2'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
