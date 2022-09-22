The code style must be the closest as possible to that recommended by Google, to find out more see [here](https://google.github.io/styleguide/pyguide.html).

## Test

Test all with

```python3
python3 test.py
```

## Code quality

To check the code quality with Pylint

```console
pylint $(git ls-files '*.py') > code-quality.txt
```

## requirements.txt

The user must be able to install the code and be able to use it through:

```console
virtualenv venv
source venv/local/bin/activate # or source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

If you add new libraries remember to put them in `requirements.txt`, keeping the file short and synthetic.

