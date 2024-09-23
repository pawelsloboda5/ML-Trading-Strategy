from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml_trading_strategy",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning-based trading strategy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ML-Trading-Strategy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "yfinance",
        "pandas",
        "numpy",
        "scikit-learn",
        "backtrader",
        "scipy",
    ],
)