from setuptools import setup
from __init__ import version

setup(name='distrend',
    version=version,
    description='Disease trend analysis platform accurately predicts the occurrence of diseases under mixed background',
    url='https://github.com/KingoftheNight/distrend',
    author='Liang YC',
    author_email='1694822092@qq.com',
    license='BSD 2-Clause',
    packages=['.'],
    install_requires=['numpy', 'pandas', 'tqdm', 'matplotlib'],
    entry_points={
        'console_scripts': [
        'distrend=distrend:distrend',
            ]
        },
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=True)
