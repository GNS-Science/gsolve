# Installation

Install from conda forge :

```bash
conda install gsolve -c conda-forge
```

OR

Install from PyPI using `pip`:

```bash
pip install gsolve
```

Note that `pygtide` internal database files may require updating. This will download the latest leap seconds and pole data.

After installation run

```bash
python -c "import pygtide; pygtide.update()"
```
