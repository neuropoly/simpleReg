from setuptools import find_packages, setup


setup(
    name="simplereg",
    version="0.1.0",
    description="3D manual registration app for NIfTI medical images",
    long_description="SimpleReg is a manual 3D registration application for medical images.",
    long_description_content_type="text/plain",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "PyQt6",
        "pyqtgraph",
        "numpy",
        "scipy",
        "nibabel",
        "scikit-image",
        "transforms3d",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "simplereg=simplereg.__main__:main",
            "simplereg_apply=simplereg.apply:main",
        ],
    },
    python_requires=">=3.10",
)
