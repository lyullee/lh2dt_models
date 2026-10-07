# Release, DOI, and PyPI Guide

## Identifiers

- Repository/project: `lh2dt_models`
- Python distribution: `lh2dt_models`
- Python implementation import: `lh2dt`
- Initial version: `0.1.0`

## Local release checks

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m build
python -m twine check dist/*
```

## GitHub

Create `lyullee/lh2dt_models` and push the `main` branch. Do not add field
historian data, photographs, videos, P&IDs, site tags, or private asset catalogs.

## DOI through Zenodo

Connect GitHub to Zenodo, enable `lyullee/lh2dt_models`, and create a GitHub
release named `v0.1.0`. Zenodo archives the release and assigns a DOI. `CITATION.cff`
and `.zenodo.json` supply the software metadata. Add the issued DOI to subsequent
release metadata rather than inventing one in source files.

## PyPI

Confirm that the distribution name is available and configure a PyPI API token or
Trusted Publisher. A local upload is:

```bash
python -m twine upload dist/*
```

The token must never be committed to Git or pasted into chat. The included GitHub
Actions workflow can publish on a release after PyPI Trusted Publisher is configured.
