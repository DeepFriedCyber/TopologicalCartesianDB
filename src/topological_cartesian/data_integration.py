#!/usr/bin/env python3
"""
Data Integration & Legacy System Connectors for TCDB

Addresses feedback on integration with existing data sources, databases, 
data lakes, and APIs. Provides seamless connectivity to enterprise systems.
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, AsyncGenerator, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

# Database connectors
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

try:
    import pymongo
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Types of data sources we can integrate with"""
    SQL_DATABASE = "sql_database"
    NOSQL_DATABASE = "nosql_database"
    DATA_LAKE = "data_lake"
    REST_API = "rest_api"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    STREAMING = "streaming"
    GRAPH_DATABASE = "graph_database"

class CubeType(Enum):
    """Target cube types for data routing"""
    CODE = "code"
    DATA = "data"
    USER = "user"
    TEMPORAL = "temporal"
    SYSTEM = "system"

@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    source_id: str
    source_type: DataSourceType
    connection_string: str
    credentials: Dict[str, str]
    target_cube: CubeType
    schema_mapping: Dict[str, str]
    refresh_interval: timedelta
    batch_size: int = 1000
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataRecord:
    """Standardized data record for cube ingestion"""
    record_id: str
    source_id: str
    cube_type: CubeType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    classification: str = "internal"
    embeddings: Optional[np.ndarray] = None

class DataConnector(ABC):
    """Abstract base class for data connectors"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.connection = None
        self.last_sync = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def extract_data(self, since: Optional[datetime] = None) -> AsyncGenerator[DataRecord, None]:
        """Extract data from source"""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection is working"""
        pass
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get schema information from data source"""
        return {"schema": "unknown"}

class SQLDatabaseConnector(DataConnector):
    """Connector for SQL databases (PostgreSQL, MySQL, SQL Server, etc.)"""
    
    async def connect(self) -> bool:
        """Connect to SQL database"""
        if not SQL_AVAILABLE:
            logger.error("SQLAlchemy not available for SQL connections")
            return False
        
        try:
            self.connection = create_engine(self.config.connection_string)
            # Test connection
            with self.connection.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"✅ Connected to SQL database: {self.config.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to SQL database {self.config.source_id}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from SQL database"""
        if self.connection:
            self.connection.dispose()
            self.connection = None
    
    async def extract_data(self, since: Optional[datetime] = None) -> AsyncGenerator[DataRecord, None]:
        """Extract data from SQL database"""
        if not self.connection:
            await self.connect()
        
        # Build query based on schema mapping
        table_name = self.config.metadata.get('table_name', 'data')
        content_column = self.config.schema_mapping.get('content', 'content')
        timestamp_column = self.config.schema_mapping.get('timestamp', 'created_at')
        
        query = f"SELECT * FROM {table_name}"
        if since:
            query += f" WHERE {timestamp_column} > '{since.isoformat()}'"
        query += f" ORDER BY {timestamp_column} LIMIT {self.config.batch_size}"
        
        try:
            with self.connection.connect() as conn:
                result = conn.execute(text(query))
                
                for row in result:
                    row_dict = dict(row._mapping)
                    
                    # Extract content
                    content = str(row_dict.get(content_column, ''))
                    
                    # Extract timestamp
                    timestamp = row_dict.get(timestamp_column, datetime.now())
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    
                    # Create metadata
                    metadata = {k: v for k, v in row_dict.items() 
                              if k not in [content_column, timestamp_column]}
                    metadata['source_table'] = table_name
                    
                    yield DataRecord(
                        record_id=str(uuid.uuid4()),
                        source_id=self.config.source_id,
                        cube_type=self.config.target_cube,
                        content=content,
                        metadata=metadata,
                        timestamp=timestamp
                    )
                    
        except Exception as e:
            logger.error(f"❌ Error extracting from SQL database {self.config.source_id}: {e}")
    
    async def validate_connection(self) -> bool:
        """Validate SQL connection"""
        try:
            with self.connection.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get SQL database schema"""
        if not self.connection:
            return {"error": "not_connected"}
        
        try:
            with self.connection.connect() as conn:
                # Get table information
                tables_query = """
                SELECT table_name, column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
                """
                result = conn.execute(text(tables_query))
                
                schema = {}
                for row in result:
                    table = row.table_name
                    if table not in schema:
                        schema[table] = {}
                    schema[table][row.column_name] = row.data_type
                
                return {"tables": schema}
                
        except Exception as e:
            return {"error": str(e)}

class NoSQLDatabaseConnector(DataConnector):
    """Connector for NoSQL databases (MongoDB, etc.)"""
    
    async def connect(self) -> bool:
        """Connect to NoSQL database"""
        if not MONGO_AVAILABLE:
            logger.error("PyMongo not available for MongoDB connections")
            return False
        
        try:
            self.connection = pymongo.MongoClient(self.config.connection_string)
            # Test connection
            self.connection.admin.command('ping')
            
            logger.info(f"✅ Connected to MongoDB: {self.config.source_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB {self.config.source_id}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from NoSQL database"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    async def extract_data(self, since: Optional[datetime] = None) -> AsyncGenerator[DataRecord, None]:
        """Extract data from MongoDB"""
        if not self.connection:
            await self.connect()
        
        db_name = self.config.metadata.get('database', 'default')
        collection_name = self.config.metadata.get('collection', 'data')
        
        db = self.connection[db_name]
        collection = db[collection_name]
        
        # Build query
        query = {}
        if since:
            timestamp_field = self.config.schema_mapping.get('timestamp', 'created_at')
            query[timestamp_field] = {"$gt": since}
        
        try:
            cursor = collection.find(query).limit(self.config.batch_size)
            
            for doc in cursor:
                # Extract content
                content_field = self.config.schema_mapping.get('content', 'content')
                content = str(doc.get(content_field, ''))
                
                # Extract timestamp
                timestamp_field = self.config.schema_mapping.get('timestamp', 'created_at')
                timestamp = doc.get(timestamp_field, datetime.now())
                
                # Create metadata
                metadata = {k: v for k, v in doc.items() 
                          if k not in [content_field, timestamp_field, '_id']}
                metadata['source_collection'] = collection_name
                metadata['document_id'] = str(doc.get('_id', ''))
                
                yield DataRecord(
                    record_id=str(uuid.uuid4()),
                    source_id=self.config.source_id,
                    cube_type=self.config.target_cube,
                    content=content,
                    metadata=metadata,
                    timestamp=timestamp
                )
                
        except Exception as e:
            logger.error(f"❌ Error extracting from MongoDB {self.config.source_id}: {e}")
    
    async def validate_connection(self) -> bool:
        """Validate MongoDB connection"""
        try:
            self.connection.admin.command('ping')
            return True
        except:
            return False

class DataLakeConnector(DataConnector):
    """Connector for data lakes (S3, Azure Data Lake, etc.)"""
    
    async def connect(self) -> bool:
        """Connect to data lake"""
        if not AWS_AVAILABLE:
            logger.error("Boto3 not available for AWS S3 connections")
            return False
        
        try:
            # Parse S3 connection string
            # Format: s3://bucket-name/prefix
            if self.config.connection_string.startswith('s3://'):
                bucket_info = self.config.connection_string[5:].split('/', 1)
                self.bucket_name = bucket_info[0]
                self.prefix = bucket_info[1] if len(bucket_info) > 1 else ''
                
                self.connection = boto3.client(
                    's3',
                    aws_access_key_id=self.config.credentials.get('access_key'),
                    aws_secret_access_key=self.config.credentials.get('secret_key'),
                    region_name=self.config.credentials.get('region', 'us-east-1')
                )
                
                # Test connection
                self.connection.head_bucket(Bucket=self.bucket_name)
                
                logger.info(f"✅ Connected to S3 data lake: {self.config.source_id}")
                return True
            else:
                logger.error(f"Unsupported data lake format: {self.config.connection_string}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to data lake {self.config.source_id}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from data lake"""
        self.connection = None
    
    async def extract_data(self, since: Optional[datetime] = None) -> AsyncGenerator[DataRecord, None]:
        """Extract data from data lake"""
        if not self.connection:
            await self.connect()
        
        try:
            # List objects in bucket
            paginator = self.connection.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    # Filter by timestamp if specified
                    if since and obj['LastModified'].replace(tzinfo=None) <= since:
                        continue
                    
                    # Download and process file
                    try:
                        response = self.connection.get_object(
                            Bucket=self.bucket_name, 
                            Key=obj['Key']
                        )
                        content = response['Body'].read().decode('utf-8')
                        
                        # Create metadata
                        metadata = {
                            'file_key': obj['Key'],
                            'file_size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'etag': obj['ETag']
                        }
                        
                        yield DataRecord(
                            record_id=str(uuid.uuid4()),
                            source_id=self.config.source_id,
                            cube_type=self.config.target_cube,
                            content=content,
                            metadata=metadata,
                            timestamp=obj['LastModified'].replace(tzinfo=None)
                        )
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process {obj['Key']}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"❌ Error extracting from data lake {self.config.source_id}: {e}")
    
    async def validate_connection(self) -> bool:
        """Validate data lake connection"""
        try:
            self.connection.head_bucket(Bucket=self.bucket_name)
            return True
        except:
            return False

class RESTAPIConnector(DataConnector):
    """Connector for REST APIs"""
    
    async def connect(self) -> bool:
        """Connect to REST API"""
        # For REST APIs, we don't maintain persistent connections
        logger.info(f"✅ REST API connector ready: {self.config.source_id}")
        return True
    
    async def disconnect(self):
        """Disconnect from REST API"""
        pass
    
    async def extract_data(self, since: Optional[datetime] = None) -> AsyncGenerator[DataRecord, None]:
        """Extract data from REST API"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                # Build API request
                url = self.config.connection_string
                headers = self.config.credentials
                
                # Add timestamp filter if supported
                params = {}
                if since and 'timestamp_param' in self.config.metadata:
                    timestamp_param = self.config.metadata['timestamp_param']
                    params[timestamp_param] = since.isoformat()
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle different API response formats
                        if isinstance(data, list):
                            items = data
                        elif isinstance(data, dict) and 'data' in data:
                            items = data['data']
                        elif isinstance(data, dict) and 'items' in data:
                            items = data['items']
                        else:
                            items = [data]
                        
                        for item in items:
                            # Extract content
                            content_field = self.config.schema_mapping.get('content', 'content')
                            content = str(item.get(content_field, json.dumps(item)))
                            
                            # Extract timestamp
                            timestamp_field = self.config.schema_mapping.get('timestamp', 'created_at')
                            timestamp_str = item.get(timestamp_field, datetime.now().isoformat())
                            
                            if isinstance(timestamp_str, str):
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            else:
                                timestamp = datetime.now()
                            
                            # Create metadata
                            metadata = {k: v for k, v in item.items() 
                                      if k not in [content_field, timestamp_field]}
                            metadata['api_endpoint'] = url
                            
                            yield DataRecord(
                                record_id=str(uuid.uuid4()),
                                source_id=self.config.source_id,
                                cube_type=self.config.target_cube,
                                content=content,
                                metadata=metadata,
                                timestamp=timestamp.replace(tzinfo=None)
                            )
                    else:
                        logger.error(f"❌ API request failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ Error extracting from REST API {self.config.source_id}: {e}")
    
    async def validate_connection(self) -> bool:
        """Validate REST API connection"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.connection_string, 
                    headers=self.config.credentials
                ) as response:
                    return response.status < 400
        except:
            return False

class DataIntegrationManager:
    """Manages all data source integrations"""
    
    def __init__(self):
        self.connectors: Dict[str, DataConnector] = {}
        self.sync_status: Dict[str, Dict[str, Any]] = {}
        self.running = False
        
    def register_data_source(self, config: DataSourceConfig) -> bool:
        """Register a new data source"""
        
        # Create appropriate connector
        connector = self._create_connector(config)
        if not connector:
            logger.error(f"❌ Failed to create connector for {config.source_id}")
            return False
        
        self.connectors[config.source_id] = connector
        self.sync_status[config.source_id] = {
            'status': 'registered',
            'last_sync': None,
            'records_synced': 0,
            'errors': []
        }
        
        logger.info(f"✅ Registered data source: {config.source_id}")
        return True
    
    def _create_connector(self, config: DataSourceConfig) -> Optional[DataConnector]:
        """Create appropriate connector based on source type"""
        
        if config.source_type == DataSourceType.SQL_DATABASE:
            return SQLDatabaseConnector(config)
        elif config.source_type == DataSourceType.NOSQL_DATABASE:
            return NoSQLDatabaseConnector(config)
        elif config.source_type == DataSourceType.DATA_LAKE:
            return DataLakeConnector(config)
        elif config.source_type == DataSourceType.REST_API:
            return RESTAPIConnector(config)
        else:
            logger.error(f"❌ Unsupported data source type: {config.source_type}")
            return None
    
    async def sync_data_source(self, source_id: str) -> bool:
        """Sync data from a specific source"""
        
        if source_id not in self.connectors:
            logger.error(f"❌ Data source not found: {source_id}")
            return False
        
        connector = self.connectors[source_id]
        status = self.sync_status[source_id]
        
        try:
            # Connect to data source
            if not await connector.connect():
                status['status'] = 'connection_failed'
                return False
            
            # Validate connection
            if not await connector.validate_connection():
                status['status'] = 'validation_failed'
                return False
            
            status['status'] = 'syncing'
            records_synced = 0
            
            # Extract data since last sync
            since = status.get('last_sync')
            
            async for record in connector.extract_data(since):
                # Process record (send to appropriate cube)
                await self._process_record(record)
                records_synced += 1
                
                # Update progress
                if records_synced % 100 == 0:
                    logger.info(f"📊 Synced {records_synced} records from {source_id}")
            
            # Update sync status
            status.update({
                'status': 'completed',
                'last_sync': datetime.now(),
                'records_synced': status['records_synced'] + records_synced,
                'last_batch_size': records_synced
            })
            
            logger.info(f"✅ Sync completed for {source_id}: {records_synced} records")
            return True
            
        except Exception as e:
            status['status'] = 'error'
            status['errors'].append(str(e))
            logger.error(f"❌ Sync failed for {source_id}: {e}")
            return False
        
        finally:
            await connector.disconnect()
    
    async def _process_record(self, record: DataRecord):
        """Process a data record and send to appropriate cube"""
        
        # Generate embeddings if needed
        if record.embeddings is None:
            record.embeddings = await self._generate_embeddings(record.content)
        
        # Route to appropriate cube based on cube_type
        await self._route_to_cube(record)
        
        logger.debug(f"📝 Processed record {record.record_id} for {record.cube_type.value} cube")
    
    async def _generate_embeddings(self, content: str) -> np.ndarray:
        """Generate embeddings for content"""
        # Placeholder - in production would use actual embedding model
        return np.random.normal(0, 1, 384)
    
    async def _route_to_cube(self, record: DataRecord):
        """Route record to appropriate cube"""
        # Placeholder - in production would integrate with actual cube system
        logger.debug(f"🎯 Routing record to {record.cube_type.value} cube")
    
    async def sync_all_sources(self):
        """Sync all registered data sources"""
        
        tasks = []
        for source_id in self.connectors.keys():
            if self.connectors[source_id].config.enabled:
                tasks.append(self.sync_data_source(source_id))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)
            logger.info(f"✅ Sync completed: {successful}/{len(tasks)} sources successful")
    
    async def start_continuous_sync(self):
        """Start continuous synchronization"""
        
        self.running = True
        logger.info("🔄 Starting continuous data synchronization")
        
        while self.running:
            try:
                await self.sync_all_sources()
                
                # Wait for next sync cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in continuous sync: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def stop_continuous_sync(self):
        """Stop continuous synchronization"""
        self.running = False
        logger.info("⏹️ Stopping continuous data synchronization")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status for all sources"""
        return {
            'sources': self.sync_status,
            'total_sources': len(self.connectors),
            'active_sources': len([s for s in self.sync_status.values() 
                                 if s['status'] == 'completed']),
            'last_update': datetime.now().isoformat()
        }

# Example configurations for common enterprise systems
def create_sample_configurations() -> List[DataSourceConfig]:
    """Create sample configurations for common enterprise systems"""
    
    configs = []
    
    # PostgreSQL database
    configs.append(DataSourceConfig(
        source_id="postgres_crm",
        source_type=DataSourceType.SQL_DATABASE,
        connection_string="postgresql://user:pass@localhost:5432/crm",
        credentials={},
        target_cube=CubeType.USER,
        schema_mapping={
            'content': 'description',
            'timestamp': 'created_at'
        },
        refresh_interval=timedelta(hours=1),
        metadata={'table_name': 'customer_interactions'}
    ))
    
    # MongoDB logs
    configs.append(DataSourceConfig(
        source_id="mongo_logs",
        source_type=DataSourceType.NOSQL_DATABASE,
        connection_string="mongodb://localhost:27017/",
        credentials={},
        target_cube=CubeType.SYSTEM,
        schema_mapping={
            'content': 'message',
            'timestamp': 'timestamp'
        },
        refresh_interval=timedelta(minutes=15),
        metadata={'database': 'logs', 'collection': 'application_logs'}
    ))
    
    # S3 data lake
    configs.append(DataSourceConfig(
        source_id="s3_documents",
        source_type=DataSourceType.DATA_LAKE,
        connection_string="s3://company-documents/processed/",
        credentials={
            'access_key': 'your_access_key',
            'secret_key': 'your_secret_key',
            'region': 'us-east-1'
        },
        target_cube=CubeType.DATA,
        schema_mapping={
            'content': 'file_content',
            'timestamp': 'last_modified'
        },
        refresh_interval=timedelta(hours=6)
    ))
    
    # REST API
    configs.append(DataSourceConfig(
        source_id="github_api",
        source_type=DataSourceType.REST_API,
        connection_string="https://api.github.com/repos/company/repo/commits",
        credentials={
            'Authorization': 'token your_github_token'
        },
        target_cube=CubeType.CODE,
        schema_mapping={
            'content': 'commit.message',
            'timestamp': 'commit.author.date'
        },
        refresh_interval=timedelta(hours=2),
        metadata={'timestamp_param': 'since'}
    ))
    
    return configs

async def test_data_integration():
    """Test data integration functionality"""
    
    print("🔗 Testing Data Integration System")
    print("=" * 50)
    
    # Initialize integration manager
    manager = DataIntegrationManager()
    
    # Create sample configurations
    configs = create_sample_configurations()
    
    # Register data sources
    for config in configs:
        success = manager.register_data_source(config)
        print(f"{'✅' if success else '❌'} Registered {config.source_id}")
    
    # Get sync status
    status = manager.get_sync_status()
    print(f"📊 Total sources registered: {status['total_sources']}")
    
    # Test individual connector (mock)
    print("\n🧪 Testing individual connectors...")
    
    # Test SQL connector (if available)
    if SQL_AVAILABLE:
        sql_config = DataSourceConfig(
            source_id="test_sql",
            source_type=DataSourceType.SQL_DATABASE,
            connection_string="sqlite:///test.db",
            credentials={},
            target_cube=CubeType.DATA,
            schema_mapping={'content': 'text', 'timestamp': 'created'},
            refresh_interval=timedelta(hours=1)
        )
        
        sql_connector = SQLDatabaseConnector(sql_config)
        print(f"✅ SQL connector created")
    
    print("\n🔗 Data Integration Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_data_integration())