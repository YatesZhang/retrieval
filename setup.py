from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="gpt_flamingo",
        packages=find_packages(),
        # include_package_data=True,
        version="0.0.1",
        license="Apache 2.0",
        description="An open-source framework for multi-modality instruction fine-tuning",
        long_description="LoRA tuning for Open Flamingo",
        long_description_content_type="text/markdown",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning"],
        install_requires=[],
    )