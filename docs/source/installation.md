# Installation

Install from PyPI using `pip`:

```bash
pip install gsolve
```

Note that `pygtide` internal database files may require updating. This will download the latest leap seconds and pole data.

```bash
python -c "import pygtide; pygtide.update()"
```