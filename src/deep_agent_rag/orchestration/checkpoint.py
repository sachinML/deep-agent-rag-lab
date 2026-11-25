
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def create_sqlite_checkpointer():
    project_root = Path(__file__).resolve().parents[3]

    database_dir = project_root / "data" / "checkpoints"
    database_dir.mkdir(parents=True, exist_ok=True)

    database_path = database_dir / "langgraph.sqlite"

    connection = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)

    return checkpointer, connection
