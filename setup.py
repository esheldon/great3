import os
import glob
from distutils.core import setup

scripts=['great3-run','great3-mean-shear','great3-calc-skynoise',
         'great3-make-condor','great3-make-wq','great3-combine-ranges']

scripts=[os.path.join('bin',s) for s in scripts]

setup(name="great3  ", 
      version="0.1.0",
      description="Run on great3",
      license = "GPL",
      author="Erin Scott Sheldon",
      author_email="erin.sheldon@gmail.com",
      scripts=scripts,
      #data_files=data_files,
      packages=['great3'])
