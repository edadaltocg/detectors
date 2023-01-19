<p align="center">
    <br>
    <img src="./face-with-monocle.svg" width="150" height="150" /> 
    <br>
</p>

# Detectors


Package to accelerate research on out-of-distribution (OOD) detection inspired by Huggingface's transformers.


Upload to PyPI
```
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*

python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-package-YOUR-USERNAME-HERE
```