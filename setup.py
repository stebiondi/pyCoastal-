from setuptools import setup, find_packages

setup(
    name='pyCoastal',
    version='0.1.0',
    author='stebiondi',
    author_email='stefano.biondi@ufl.edu',
    description='A Python module for Coastal Engineering',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    url='https://github.com/yourusername/pyCoastal',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
