"""
SQL Database Service
Manages PostgreSQL connection and schema introspection for structured medical data.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, inspect, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class SQLDatabaseService:
    """
    Service for managing PostgreSQL connection and providing schema information.
    
    Supports:
    - Connection pooling with SQLAlchemy
    - Schema introspection
    - Table and column metadata extraction
    - Sample data retrieval for LLM context
    """
    
    def __init__(self):
        """Initialize SQL database connection."""
        load_dotenv()
        
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "9000")
        self.db_name = os.getenv("DB_NAME", "medlearn_collection")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "dba")
        
        self.engine = None
        self.metadata = None
        self.session_maker = None
        
        logger.info("SQLDatabaseService initialized with connection params")
    
    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create connection string
            connection_string = (
                f"postgresql://{self.db_user}:{self.db_password}@"
                f"{self.db_host}:{self.db_port}/{self.db_name}"
            )
            
            logger.debug(f"Attempting connection to {self.db_host}:{self.db_port}/{self.db_name}")
            
            # Create engine with connection pooling
            self.engine = create_engine(
                connection_string,
                echo=False,  # Set to True for SQL debug logging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("✓ PostgreSQL connection established successfully")
            
            # Load metadata
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            
            # Create session factory
            self.session_maker = sessionmaker(bind=self.engine)
            
            logger.info(f"✓ Loaded schema with {len(self.metadata.tables)} tables")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
            return False
    
    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as list of dictionaries.
        
        Args:
            sql_query: SQL query string
            
        Returns:
            List of result rows as dictionaries
            
        Raises:
            Exception if query fails
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            logger.debug(f"Executing SQL query: {sql_query[:100]}...")
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                
                # Get column names
                columns = result.keys()
                
                # Convert to list of dictionaries
                rows = []
                for row in result:
                    row_dict = dict(zip(columns, row))
                    rows.append(row_dict)
                
                logger.debug(f"Query returned {len(rows)} rows")
                return rows
                
        except Exception as e:
            logger.error(f"SQL query execution failed: {e}", exc_info=True)
            raise
    
    def get_schema_info(self) -> str:
        """
        Get schema information formatted for LLM context.
        
        Returns:
            Formatted schema description
        """
        if not self.metadata:
            return "No schema information available"
        
        schema_info = "Medical Database Schema:\n\n"
        
        for table_name, table in self.metadata.tables.items():
            schema_info += f"Table: {table_name}\n"
            schema_info += f"Columns:\n"
            
            for column in table.columns:
                col_type = str(column.type)
                nullable = "nullable" if column.nullable else "not null"
                schema_info += f"  - {column.name} ({col_type}, {nullable})\n"
            
            schema_info += "\n"
        
        return schema_info
    
    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in the database.
        
        Returns:
            List of table names
        """
        if not self.metadata:
            return []
        
        return list(self.metadata.tables.keys())
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table info or None if table not found
        """
        if not self.metadata or table_name not in self.metadata.tables:
            return None
        
        table = self.metadata.tables[table_name]
        
        columns = []
        for col in table.columns:
            columns.append({
                "name": col.name,
                "type": str(col.type),
                "nullable": col.nullable,
                "primary_key": col.primary_key,
            })
        
        return {
            "name": table_name,
            "columns": columns,
            "column_count": len(columns),
        }
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get sample data from a table for LLM context.
        
        Args:
            table_name: Name of the table
            limit: Number of rows to retrieve
            
        Returns:
            List of sample rows
        """
        if not self.engine or table_name not in self.metadata.tables:
            return []
        
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return self.execute_query(query)
        except Exception as e:
            logger.warning(f"Failed to get sample data from {table_name}: {e}")
            return []
    
    def validate_table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        if not self.metadata:
            return False
        
        return table_name in self.metadata.tables
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global instance
_sql_db_service: Optional[SQLDatabaseService] = None


def get_sql_database() -> SQLDatabaseService:
    """Get or create the global SQL database service instance."""
    global _sql_db_service
    
    if _sql_db_service is None:
        _sql_db_service = SQLDatabaseService()
        if not _sql_db_service.connect():
            logger.error("Failed to initialize SQL database service")
            raise RuntimeError("Cannot connect to PostgreSQL database")
    
    return _sql_db_service
