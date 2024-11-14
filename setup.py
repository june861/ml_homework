# -*- encoding: utf-8 -*-
'''
@File    :   setup.py
@Time    :   2024/11/10 15:30:23
@Author  :   junewluo 
'''
import setuptools
from pathlib import Path
from setuptools import setup, find_packages

# 读取 requirements.txt 文件
def read_requirements():
    req_path = Path(__file__).parent / 'requirements.txt'
    with req_path.open() as f:
        requirements = f.read().splitlines()
    return requirements

# 读取 README.md 文件作为 long_description
def read_readme():
    readme_path = Path(__file__).parent / 'README.md'
    with readme_path.open(encoding='utf-8') as f:
        return f.read()

setup(
    name='ml_homework',
    author="junweiluo",
    version='0.1',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=read_requirements(),
)