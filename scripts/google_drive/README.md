# Google Drive Functionality

## Important Notice

The main Google Drive functionality is implemented in:

```
src/utils/google_drive_manager.py
```

This is the authoritative implementation and should be the only one maintained.

## Files in this Directory

- `google_drive_manager.py` - A redirect module that imports from the main implementation
- Other files in this directory should be considered deprecated

## Usage

Always import directly from the main implementation:

```python
from src.utils.google_drive_manager import sync_to_drive, configure_sync_method
```

Or use the redirect if you're maintaining backward compatibility:

```python
from scripts.google_drive.google_drive_manager import sync_to_drive, configure_sync_method
```

## Deprecated Files

The following files are deprecated and should not be modified or used:

- `google_drive_manager_impl.py`
- `ensure_drive_manager.py`

Instead, make changes directly to `src/utils/google_drive_manager.py`.
