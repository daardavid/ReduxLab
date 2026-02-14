"""
Data Connectors Module for ReduxLab

Provides unified data loading from multiple sources:
- File formats: Excel, CSV, Parquet, JSON, Feather, Stata, SPSS, SAS
- Databases: SQLite, PostgreSQL, MySQL/MariaDB, SQL Server

All loaders return Dict[str, pd.DataFrame] where keys are sheet/table names.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-based loaders
# ---------------------------------------------------------------------------

def load_excel(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Load all sheets from an Excel file."""
    logger.info(f"Loading Excel file: {path}")
    data = pd.read_excel(path, sheet_name=None, **kwargs)
    logger.info(f"Loaded {len(data)} sheets from Excel")
    return data


def load_csv(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Load a CSV file. Returns dict with filename stem as key."""
    logger.info(f"Loading CSV file: {path}")
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            name = Path(path).stem
            logger.info(f"Loaded CSV with encoding {enc}: {df.shape}")
            return {name: df}
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not decode CSV file with any supported encoding: {path}")


def load_parquet(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Load a Parquet file."""
    logger.info(f"Loading Parquet file: {path}")
    df = pd.read_parquet(path, **kwargs)
    name = Path(path).stem
    logger.info(f"Loaded Parquet: {df.shape}")
    return {name: df}


def load_json(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Load a JSON or JSON Lines file."""
    logger.info(f"Loading JSON file: {path}")
    try:
        df = pd.read_json(path, **kwargs)
    except ValueError:
        df = pd.read_json(path, lines=True, **kwargs)
    name = Path(path).stem
    logger.info(f"Loaded JSON: {df.shape}")
    return {name: df}


def load_feather(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Load a Feather/Arrow IPC file."""
    logger.info(f"Loading Feather file: {path}")
    df = pd.read_feather(path, **kwargs)
    name = Path(path).stem
    logger.info(f"Loaded Feather: {df.shape}")
    return {name: df}


def load_stata(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Load a Stata .dta file."""
    logger.info(f"Loading Stata file: {path}")
    df = pd.read_stata(path, **kwargs)
    name = Path(path).stem
    logger.info(f"Loaded Stata: {df.shape}")
    return {name: df}


def load_spss(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Load an SPSS .sav file."""
    logger.info(f"Loading SPSS file: {path}")
    df = pd.read_spss(path, **kwargs)
    name = Path(path).stem
    logger.info(f"Loaded SPSS: {df.shape}")
    return {name: df}


def load_sas(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Load a SAS .sas7bdat or .xpt file."""
    logger.info(f"Loading SAS file: {path}")
    df = pd.read_sas(path, **kwargs)
    name = Path(path).stem
    logger.info(f"Loaded SAS: {df.shape}")
    return {name: df}


# ---------------------------------------------------------------------------
# Database loaders
# ---------------------------------------------------------------------------

def load_sqlite(path: str, table: Optional[str] = None,
                query: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load data from a SQLite / .db file.

    Args:
        path: Path to the SQLite database file.
        table: Table name to load. If None, loads all tables.
        query: Custom SQL query. Overrides table if provided.

    Returns:
        Dict mapping table/query names to DataFrames.
    """
    import sqlite3
    logger.info(f"Loading SQLite database: {path}")

    conn = sqlite3.connect(path)
    try:
        if query:
            df = pd.read_sql_query(query, conn)
            return {"query_result": df}

        if table:
            df = pd.read_sql_table(table, f"sqlite:///{path}")
            return {table: df}

        # Load all tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row[0] for row in cursor.fetchall()]
        result = {}
        for t in tables:
            result[t] = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
            logger.info(f"  Table '{t}': {result[t].shape}")
        logger.info(f"Loaded {len(result)} tables from SQLite")
        return result
    finally:
        conn.close()


def load_sql(connection_string: str, table: Optional[str] = None,
             query: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load data from a SQL database using SQLAlchemy.

    Supports PostgreSQL, MySQL/MariaDB, SQL Server, and any SQLAlchemy-compatible engine.

    Args:
        connection_string: SQLAlchemy connection string, e.g.:
            - postgresql://user:pass@host:5432/dbname
            - mysql+pymysql://user:pass@host:3306/dbname
            - mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server
        table: Table name to load. If None and no query, lists available tables.
        query: Custom SQL query. Overrides table if provided.

    Returns:
        Dict mapping table/query names to DataFrames.
    """
    try:
        from sqlalchemy import create_engine, inspect
    except ImportError:
        raise ImportError(
            "SQLAlchemy is required for database connections. "
            "Install it with: pip install sqlalchemy"
        )

    logger.info(f"Connecting to SQL database...")
    engine = create_engine(connection_string)

    try:
        if query:
            df = pd.read_sql_query(query, engine)
            logger.info(f"Query result: {df.shape}")
            return {"query_result": df}

        if table:
            df = pd.read_sql_table(table, engine)
            logger.info(f"Table '{table}': {df.shape}")
            return {table: df}

        # Load all tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        result = {}
        for t in tables:
            result[t] = pd.read_sql_table(t, engine)
            logger.info(f"  Table '{t}': {result[t].shape}")
        logger.info(f"Loaded {len(result)} tables from database")
        return result
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

# Supported file extensions and their loaders
FILE_LOADERS = {
    '.xlsx': load_excel,
    '.xls': load_excel,
    '.csv': load_csv,
    '.parquet': load_parquet,
    '.json': load_json,
    '.jsonl': load_json,
    '.feather': load_feather,
    '.arrow': load_feather,
    '.dta': load_stata,
    '.sav': load_spss,
    '.sas7bdat': load_sas,
    '.xpt': load_sas,
    '.db': load_sqlite,
    '.sqlite': load_sqlite,
    '.sqlite3': load_sqlite,
}

SUPPORTED_EXTENSIONS = list(FILE_LOADERS.keys())

# File type filter for dialogs
FILE_TYPE_FILTERS = [
    ("Excel files", "*.xlsx *.xls"),
    ("CSV files", "*.csv"),
    ("Parquet files", "*.parquet"),
    ("JSON files", "*.json *.jsonl"),
    ("Feather/Arrow files", "*.feather *.arrow"),
    ("Stata files", "*.dta"),
    ("SPSS files", "*.sav"),
    ("SAS files", "*.sas7bdat *.xpt"),
    ("SQLite databases", "*.db *.sqlite *.sqlite3"),
    ("All files", "*.*"),
]


def load_file(path: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Unified file loader that detects format from extension.

    Args:
        path: Path to the data file.
        **kwargs: Additional arguments passed to the specific loader.

    Returns:
        Dict mapping sheet/table names to DataFrames.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = Path(path).suffix.lower()
    loader = FILE_LOADERS.get(ext)

    if loader is None:
        supported = ', '.join(SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file format: '{ext}'. Supported: {supported}"
        )

    logger.info(f"Auto-detected format: {ext} -> {loader.__name__}")
    return loader(path, **kwargs)


def get_database_tables(connection_string: str) -> List[str]:
    """
    List available tables in a SQL database.

    Args:
        connection_string: SQLAlchemy connection string.

    Returns:
        List of table names.
    """
    try:
        from sqlalchemy import create_engine, inspect
    except ImportError:
        raise ImportError("SQLAlchemy is required. Install with: pip install sqlalchemy")

    engine = create_engine(connection_string)
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    finally:
        engine.dispose()
