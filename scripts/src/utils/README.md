# Utilities for Gen-Text-AI

This directory contains utility modules for the Gen-Text-AI project.

## Google Drive Functionality

The `google_drive_manager.py` file in this directory is a **redirect** to the main implementation in:

```
src/utils/google_drive_manager.py
```

This is the authoritative implementation and should be the only one maintained.

## Directory Structure

The project maintains utilities in the following locations:

- **Main Utilities**: `src/utils/` - This is the primary location for utility modules
- **Script Utilities**: `scripts/utilities/` - Contains standalone utility scripts
- **Legacy Structure**: `scripts/src/utils/` - This directory (redirects to main utilities)

## Best Practices

Always import directly from the main utils package:

```python
from src.utils.google_drive_manager import sync_to_drive
```

This directory exists only for backward compatibility with existing imports.
