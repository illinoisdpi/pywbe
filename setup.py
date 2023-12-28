from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='pyWBE',
  version='0.0.4',
  description='A Python Library for wastewater-based epidemiology.',
  long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
  long_description_content_type="text/markdown",
  url='',  
  author='Anuj Tiwari',
  author_email='anujt@uic.edu',
  license='MIT', 
  classifiers=classifiers,
  keywords='Python_Library', 
  packages=find_packages('src', exclude=['test']),
  install_requires=[''] 
)

if __name__ == "__main__":
    setup()
