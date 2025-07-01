from setuptools import setup, find_packages

setup(
    name="diabetes_detector",
    version="0.1.0",
    description="Diyabet tahmini için Streamlit tabanlı veri bilimi projesi",
    author="Ömer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
        "matplotlib",
        "seaborn",
        "missingno"
    ],
    include_package_data=True,
    python_requires=">=3.8"
)