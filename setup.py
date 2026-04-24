from setuptools import setup, find_packages

setup(
    name='torcwa',
    version='0.2.0.dev1',
    description='GPU-accelerated Fourier modal method with automatic differentiation',
    author='Changhyun Kim',
    author_email='kch3782@snu.ac.kr',
    license='LGPL',
    url='https://github.com/kch3782/torcwa',
    install_requires=['torch>=2.11'],
    packages=find_packages(),
    keywords='torcwa',
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ]
)
