from setuptools import setup, find_packages

setup(
    name="rare-unet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision", 
        "torchaudio",
        "numpy",
        "wandb",
        "torchio", #Check
        "tqdm",
        "pytorch-cuda",
        "matplotlib",
        "nibabel",
        "omegaconf",
        "hydra-core",
    ],
    python_requires=">=3.8",# CHECK
)