"""
Base adapter for cube components in the TopologicalCartesianDB system.
"""
from typing import Any, List, Dict, Tuple, Set, Optional

class CubeAdapter:
    """
    Base class for all cubes to ensure consistent interface.
    
    This adapter defines the core interface that all cube components should
    implement or pass through to their source database.
    """
    
    def __init__(self, source_db: Any):
        """
        Initialize the adapter with a source database.
        
        Args:
            source_db: The source database or adapter to wrap
        
        Raises:
            AttributeError: If source_db lacks required methods
        """
        self.source = source_db
        self._validate_source()
    
    def _validate_source(self) -> None:
        """
        Ensure source has the required methods.
        
        Raises:
            AttributeError: If a required method is missing
        """
        required_methods = ['insert', 'query']
        for method in required_methods:
            if not hasattr(self.source, method):
                raise AttributeError(f"Source database missing required method: {method}")
    
    # Default implementations that pass through to the source
    
    def insert(self, *args, **kwargs) -> Any:
        """Pass through to source database insert method."""
        return self.source.insert(*args, **kwargs)
    
    def query(self, *args, **kwargs) -> Any:
        """Pass through to source database query method."""
        return self.source.query(*args, **kwargs)
    
    # Optional pass-through methods
    
    def clear(self) -> None:
        """Clear the database if supported by source."""
        if hasattr(self.source, 'clear'):
            self.source.clear()
        else:
            raise NotImplementedError("Clear method not supported by source database")
    
    def optimize(self) -> Dict[str, Any]:
        """Optimize the database if supported by source."""
        if hasattr(self.source, 'optimize'):
            return self.source.optimize()
        else:
            raise NotImplementedError("Optimize method not supported by source database")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics if supported by source."""
        if hasattr(self.source, 'get_stats'):
            base_stats = self.source.get_stats()
            # Add adapter info
            base_stats['adapter_type'] = self.__class__.__name__
            return base_stats
        else:
            return {'adapter_type': self.__class__.__name__}
    
    @property
    def __class_hierarchy__(self) -> List[str]:
        """
        Get the class hierarchy of the cube stack.
        Useful for debugging and understanding the cube composition.
        
        Returns:
            List of class names from the outermost to innermost cube
        """
        hierarchy = [self.__class__.__name__]
        current = self.source
        
        # Follow the chain of cubes to the core
        while hasattr(current, '__class__') and hasattr(current, 'source'):
            hierarchy.append(current.__class__.__name__)
            current = getattr(current, 'source', None)
        
        # Add the core class if available
        if current and hasattr(current, '__class__'):
            hierarchy.append(current.__class__.__name__)
            
        return hierarchy
