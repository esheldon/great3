import os
import glob
from distutils.core import setup

#scripts=['nsim-average-outputs']

#scripts=[os.path.join('bin',s) for s in scripts]

#conf_files=glob.glob('config/*.yaml')

#data_files=[]
#for f in conf_files:
#    data_files.append( ('share/nsim_config',[f]) )


setup(name="great3  ", 
      version="0.1.0",
      description="Run on great3",
      license = "GPL",
      author="Erin Scott Sheldon",
      author_email="erin.sheldon@gmail.com",
      #scripts=scripts,
      #data_files=data_files,
      packages=['great3'])
