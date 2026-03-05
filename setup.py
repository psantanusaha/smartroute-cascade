from setuptools import setup, find_packages

setup(
    name="smartroute-cascade",
    version="0.1.0",
    author="psantanusaha",
    description="A training-free semantic router for LLM cascades (Groq, Anthropic, OpenAI).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/psantanusaha/smartroute-cascade",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.30.0",
        "python-dotenv>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
