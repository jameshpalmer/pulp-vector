from setuptools import setup

setup(
    name="pulp_vector",
    version="0.1.0",
    description="Python classes for simplifying vector-based linear programming operations using the PuLP and NumPy libraries",
    author="James Palmer",
    author_email="jameshpalmer0@gmail.com",
    packages=["pulp_vector"],
    install_requires=[
        "pulp",
        "numpy",
        "pandas"
    ],
)
