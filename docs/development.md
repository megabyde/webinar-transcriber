# Development

## Toolchain

- Python 3.12
- `uv`
- `ruff`
- `ty`
- `pytest`
- `pytest-cov`

## Native Dependencies

For local development on macOS, use Homebrew:

```bash
brew install ffmpeg
```

Ubuntu CI uses platform-native packages to keep runner setup simple.

## Verification

Use the full verification cycle before each logical commit:

```bash
make format
make lint
make typecheck
make test
make coverage
```

The repository keeps tiny committed media fixtures so the pipeline tests can run without network
fetches or giant binary blobs.
