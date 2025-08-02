# setup.py
from setuptools import setup, find_packages

setup(
    name='tabulapy',
    version='1.0.0',
    author='Your Name',
    author_email='simonwa01@gmail.com',
    description='A sophisticated library for generating complex LaTeX tables.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/simonsejse/tabulapy', # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing :: Markup :: LaTeX',
    ],
    python_requires='>=3.7',
)