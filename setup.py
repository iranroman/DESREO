from setuptools import setup                                                                                 
import imp                                                                                                   
                                                                                                             
                                                                                                             
with open('README.md') as file:                                                                              
    long_description = file.read()                                                                           
                                                                                                             
version = imp.load_source('desreo.version', 'desreo/version.py')                         
                                                                                                             
setup(                                                                                                       
    name='',                                                                                 
    version=version.version,                                                                                 
    description='',
    author='',                                                                                  
    author_email='iran@ccrma-gate.stanford.edu',                                                             
    url='',                                                                                                  
    download_url='https://github.com/iranroman/desreo',                                            
    packages=['desreo'],                                                                           
    long_description=long_description,                                                                       
    long_description_content_type='text/markdown',                                                           
    keywords='',                                              
    license='',                                                                  
    classifiers=[                                                                                            
            "License :: Creative Commons Attribution 4.0",                                                   
            "Programming Language :: Python",                                                                
            "Development Status :: 3 - Alpha",                                                               
            "Intended Audience :: Developers",                                                               
            "Intended Audience :: Science/Research",                                                         
            "Topic :: Multimedia :: Audio :: Rap",                                            
        ],                                                                                                   
    install_requires=[
        'pathlib',
		'pytube',
        'demucs',
    ],                                                                                                       
)    
