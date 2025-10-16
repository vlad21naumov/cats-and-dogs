# Sphinx

[Documentation](https://www.sphinx-doc.org/en/master/index.html)

**Sphinx** - documentation generator.

To use it you need to write docstrings for your classes and functions.master

## 1. Installation

You can install it via pip, conda or poetry. Using poetry

```
uv add --dev sphinx --active
```

## 2. Launching

You need to create `docs` dir and start **sphinx** in it:

```
mkdir docs
cd docs
sphinx-quickstart
```

## 3. Structure

- `conf.py` - main configuration file with settings for documentation
- `index.rst` - documentation table of contents
- `Makefile`, `make.bat` - files to generate documentation (don't edit them!)

Edit `conf.py`:

1. Change the os.path.abspath path to the appropriate location for the code to
   be documented (from the conf.py file):

```
sys.path.insert(0, os.path.abspath("<path-to-dir>"))
```

2. Add extensions

```
extensions = [
    "<extension-1>",
    "<extension-2>",
    ...
]
```

3. Change theme

```
html_theme = "<theme>"
```

4. Generate .rst files of package:

```
sphinx-apidoc -f -o <output_dir> <module_dir>
```

5. Add files to `index.rst`:

```
.. toctree::
   :maxdepth: 4
   :caption: Contents:

   <module-1>
   <module-2>
   <module-3>
   ...
```

6. Make HTML:

```
make html
```
