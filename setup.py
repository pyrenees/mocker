#!/usr/bin/env python
from setuptools import setup

import versiontools_support


setup(
    name="mocker",
    version=":versiontools:mocker",
    description="Graceful platform for test doubles in Python (mocks, stubs, fakes, and dummies).",
    author="Gustavo Niemeyer",
    author_email="gustavo@niemeyer.net",
    maintainer="Zygmunt Krynicki",
    maintainer_email="zygmunt.krynicki@linaro.org",
    license="BSD",
    url="http://labix.org/mocker",
    download_url="https://launchpad.net/mocker/+download",
    py_modules=["mocker"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Python Software Foundation License",
        "Programming Language :: Python",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
