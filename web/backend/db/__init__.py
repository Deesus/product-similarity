"""
Please run the Jupyter notebook cells that generate the DB -- it needs to be created in parallel with Annoy index
generation. It essentially means rerunning the entire notebook, so there's no point in creating a separate Python
module specifically to generate it.

For reference, table schema is as follows:
```
CREATE TABLE products (
    id INTEGER PRIMARY KEY UNIQUE,
    file_path TEXT NOT NULL
);
```
"""

from .db import get_db_connection
