import setuptools
setuptools.setup(
    name="bcg",
    version="0.1.0",
    url="https://github.com/pokutta/bcg",
    author="(see README.md)",
    author_email="(see README.md)",
    description="Blended Conditional Gradient (BCG) Algorithm Package in Python",
    packages=setuptools.find_packages(),
    install_requires=["numpy","tableprint","tabulate"],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)

#     long_description=open('README.rst').read(),