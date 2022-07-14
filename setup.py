
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ires',
    version='0.0.1',
    author='Trevor Peyton',
    description='Ionizing Radiation Effects Spectroscopy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/TrevorPeyton/ires',
    project_urls = {
        "Bug Tracker": "https://github.com/TrevorPeyton/ires/issues"
    },
    packages=['ires'],
    install_requires=['requests'],
)