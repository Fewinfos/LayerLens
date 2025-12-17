from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="layerlens",
    version="0.1.1",
    author="navi-04",
    author_email="naveenrajthiyagarajan6@gmail.com",
    description="A library for layer-by-layer explainability of deep learning models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/navi-04/layerlens",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
)
