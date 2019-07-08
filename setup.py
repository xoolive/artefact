from setuptools import setup, find_packages

setup(
    name="artefact",
    version="0.1",
    author="Benoit Viry",
    url="https://github.com/ViryBe/artefact/",
    description="AutoencodeR TsnE For anomaly detection in AirCraft Trajectories",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sklearn",
        "torch",
        "traffic",
        "matplotlib",
        "scipy",
    ],
    python_requires=">=3.6",
)
