from setuptools import setup

setup(
    name='DNN',
    version='1.0.0',    
    description='A DNN package using numpy to implement which currently include FNN',
    url='https://github.com/AlbertDoggyLin/FeedforwardNeuralNetwork/tree/main/Package#egg=DNN',
    author='Albert Lin',
    author_email='albertlin2468@gmail.com',
    license='BSD 2-clause',
    packages=['DNN'],
    install_requires=['numpy'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)