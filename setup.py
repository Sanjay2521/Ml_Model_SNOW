"""Setup script for ML Model SNOW"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-model-snow",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end ML solution for ServiceNow incident auto-assignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Ml_Model_SNOW",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ml-snow-train=scripts.train:main",
            "ml-snow-predict=scripts.predict:main",
            "ml-snow-evaluate=scripts.evaluate:main",
            "ml-snow-deploy=scripts.deploy:main",
        ],
    },
)
