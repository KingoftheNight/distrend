from setuptools import setup
from dtap.__init__ import version

setup(name='dtap',
    version=version,
    description='Disease trend analysis platform accurately predicts the occurrence of diseases under mixed background',
    url='https://github.com/KingoftheNight/Dtap',
    author='Liang YC',
    author_email='1694822092@qq.com',
    license='BSD 2-Clause',
    packages=['dtap'],
    install_requires=['shap', 'numpy', 'scikit-learn', 'pandas', 'xgboost', 'skrebate', 'matplotlib', 'ipython'],
    entry_points={
        'console_scripts': [
        'dtap=dtap.Dtap:Dtap',
            ]
        },
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=True)
