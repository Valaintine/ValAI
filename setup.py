from setuptools import setup, find_packages 
 
with open("README.md", "r", encoding="utf-8") as fh: 
    long_description = fh.read() 
 
setup( 
    name="persona-chatbot-sdk", 
    version="0.1.0", 
    author="Your Name", 
    author_email="your.email@example.com", 
    description="A powerful SDK for creating personality-driven chatbots", 
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    packages=find_packages(), 
    classifiers=[ 
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent", 
    ], 
    python_requires=">=3.8", 
) 
