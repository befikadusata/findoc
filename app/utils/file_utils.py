import os
from werkzeug.utils import secure_filename

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename to prevent path traversal issues and ensure it's safe for storage.
    Uses werkzeug.utils.secure_filename to create a safe version of the filename.
    """
    # Ensure the filename is treated as a basename, removing any path components
    basename = os.path.basename(filename)
    # Secure the filename to prevent malicious input
    return secure_filename(basename)
