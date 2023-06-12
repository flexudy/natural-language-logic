from distutils.core import setup
import setuptools

setup(
    name='natural-language-logic',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://flexudy.com',
    license='',
    author='Flexudy',
    author_email='support@flexudy.com',
    description='Natural Language Logic',
    install_requires=["pytest==6.2.1",
                      "pytest-cov==2.10.1",
                      "numpy~=1.19.4",
                      "torch==1.9.0",
                      "transformers==4.30.0",
                      "sentencepiece==0.1.95",
                      "protobuf==3.15.3",
                      "langdetect==1.0.9"]
)
