# Contributing to NeuroTK

Thank you for your interest in contributing to NeuroTK. This document describes how to
set up a development environment, run the test suite, and submit changes.

## Getting Started

1. Fork the repository on GitHub and clone your fork:

   ```sh
   git clone https://github.com/<your-username>/neurotk.git
   cd neurotk
   ```

2. Create and activate a virtual environment:

   ```sh
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. Install the package in editable mode with all extras:

   ```sh
   pip install -e ".[inference]"
   pip install pytest
   ```

## Running Tests

The test suite uses pytest and runs against synthetic NIfTI fixtures generated at
test time (no external data required):

```sh
pytest tests/ -v
```

All tests must pass before submitting a pull request. The CI pipeline
(`.github/workflows/ci.yml`) runs the full suite on Python 3.8, 3.9, 3.10, and 3.11.

## Code Style

- Follow PEP 8.
- Use type annotations for function signatures.
- Keep functions focused and well-named; prefer clarity over brevity.

## Submitting Changes

1. Create a feature branch from `main`:

   ```sh
   git checkout -b feature/my-change
   ```

2. Make your changes and add tests covering the new functionality.

3. Run the test suite locally and confirm all tests pass.

4. Commit with a clear message describing the change:

   ```sh
   git commit -m "Add support for DICOM input validation"
   ```

5. Push your branch and open a pull request against `main`.

## Reporting Bugs

Please file bug reports using the
[Bug Report template](https://github.com/SakshiRa/neurotk/issues/new?template=bug_report.md)
on GitHub Issues. Include:

- Python version and operating system
- NeuroTK version (`neurotk --version` or `python -c "import neurotk; print(neurotk.__version__)"`)
- Minimal steps to reproduce the issue
- Relevant error messages or logs

## Requesting Features

Use the
[Feature Request template](https://github.com/SakshiRa/neurotk/issues/new?template=feature_request.md)
on GitHub Issues to suggest new functionality.

## Support

- **GitHub Issues**: https://github.com/SakshiRa/neurotk/issues
- **Maintainer**: Sakshi Rathi (rathi036@umn.edu)
- Issues are monitored regularly. Bug reports are typically acknowledged within one week.
  Feature requests and pull requests are reviewed on a best-effort basis.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to abide by its terms.
