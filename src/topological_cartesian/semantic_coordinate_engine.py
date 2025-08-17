#!/usr/bin/env python3
"""
Semantic Coordinate Engine

Addresses the core flaw identified in feedback: replaces heuristic keyword-based
coordinate generation with proper semantic embedding and learned coordinate mapping.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import pickle
import hashlib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Import embedding models with fallbacks
EMBEDDING_BACKENDS_AVAILABLE = {}

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_BACKENDS_AVAILABLE['sentence_transformers'] = True
    logger.info("✅ SentenceTransformers available")
except ImportError:
    EMBEDDING_BACKENDS_AVAILABLE['sentence_transformers'] = False
    logger.warning("❌ SentenceTransformers not available")

try:
    import openai
    EMBEDDING_BACKENDS_AVAILABLE['openai'] = True
    logger.info("✅ OpenAI embeddings available")
except ImportError:
    EMBEDDING_BACKENDS_AVAILABLE['openai'] = False
    logger.warning("❌ OpenAI embeddings not available")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    EMBEDDING_BACKENDS_AVAILABLE['transformers'] = True
    logger.info("✅ HuggingFace Transformers available")
except ImportError:
    EMBEDDING_BACKENDS_AVAILABLE['transformers'] = False
    logger.warning("❌ HuggingFace Transformers not available")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    EMBEDDING_BACKENDS_AVAILABLE['sklearn'] = True
    logger.info("✅ Scikit-learn available (fallback)")
except ImportError:
    EMBEDDING_BACKENDS_AVAILABLE['sklearn'] = False
    logger.warning("❌ Scikit-learn not available")


@dataclass
class SemanticCoordinateMapping:
    """Learned mapping from semantic embeddings to interpretable coordinates"""
    dimension_name: str
    embedding_to_coord_model: Any  # Trained model (e.g., neural network, regression)
    training_examples: List[Tuple[str, float]] = field(default_factory=list)
    validation_score: float = 0.0
    model_type: str = "unknown"
    feature_importance: Optional[Dict[str, float]] = None


class SemanticEmbedder(ABC):
    """Abstract base class for semantic embedding generation"""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings for batch of texts"""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder"""
        pass


class SentenceTransformerEmbedder(SemanticEmbedder):
    """Semantic embedder using SentenceTransformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if not EMBEDDING_BACKENDS_AVAILABLE['sentence_transformers']:
            raise ImportError("SentenceTransformers not available")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Initialized SentenceTransformer: {model_name}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text"""
        return self.model.encode([text])[0]
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings for batch of texts"""
        return self.model.encode(texts)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()


class TransformerEmbedder(SemanticEmbedder):
    """Semantic embedder using HuggingFace Transformers"""
    
    def __init__(self, model_name: str = 'microsoft/DialoGPT-medium'):
        if not EMBEDDING_BACKENDS_AVAILABLE['transformers']:
            raise ImportError("Transformers not available")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self._embedding_dim = self.model.config.hidden_size
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized Transformer: {model_name}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()[0]
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings for batch of texts"""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self._embedding_dim


class TfidfEmbedder(SemanticEmbedder):
    """Fallback semantic embedder using TF-IDF + PCA"""
    
    def __init__(self, max_features: int = 1000, n_components: int = 384):
        if not EMBEDDING_BACKENDS_AVAILABLE['sklearn']:
            raise ImportError("Scikit-learn not available")
        
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
        self.is_fitted = False
        
        logger.info(f"Initialized TF-IDF embedder with {n_components} dimensions")
    
    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer and PCA on a corpus"""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.pca.fit(tfidf_matrix.toarray())
        self.is_fitted = True
        logger.info(f"TF-IDF embedder fitted on {len(texts)} texts")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text"""
        if not self.is_fitted:
            # Fit on the single text (not ideal, but necessary for fallback)
            self.fit([text])
        
        tfidf_vector = self.vectorizer.transform([text])
        embedding = self.pca.transform(tfidf_vector.toarray())
        return embedding[0]
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings for batch of texts"""
        if not self.is_fitted:
            self.fit(texts)
        
        tfidf_matrix = self.vectorizer.transform(texts)
        embeddings = self.pca.transform(tfidf_matrix.toarray())
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.n_components


class CoordinateMapper:
    """Maps semantic embeddings to interpretable coordinates using learned models"""
    
    def __init__(self):
        self.dimension_mappers = {}
        self.training_data = {}
        
        # Import ML models for coordinate mapping
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            from sklearn.neural_network import MLPRegressor
            self.ml_available = True
        except ImportError:
            self.ml_available = False
            logger.warning("Scikit-learn not available for coordinate mapping")
    
    def add_training_example(self, text: str, coordinates: Dict[str, float]):
        """Add a training example for coordinate mapping"""
        for dim_name, coord_value in coordinates.items():
            if dim_name not in self.training_data:
                self.training_data[dim_name] = []
            
            self.training_data[dim_name].append((text, coord_value))
    
    def train_coordinate_mappers(self, embedder: SemanticEmbedder):
        """Train coordinate mappers using collected training data"""
        if not self.ml_available:
            logger.error("Cannot train coordinate mappers: scikit-learn not available")
            return
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        for dim_name, training_examples in self.training_data.items():
            if len(training_examples) < 10:
                logger.warning(f"Insufficient training data for dimension {dim_name}: {len(training_examples)} examples")
                continue
            
            # Prepare training data
            texts = [example[0] for example in training_examples]
            coordinates = [example[1] for example in training_examples]
            
            # Generate embeddings
            embeddings = embedder.embed_batch(texts)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, coordinates, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store mapping
            mapping = SemanticCoordinateMapping(
                dimension_name=dim_name,
                embedding_to_coord_model=model,
                training_examples=training_examples,
                validation_score=r2,
                model_type="RandomForestRegressor",
                feature_importance=dict(zip(
                    [f"emb_{i}" for i in range(embeddings.shape[1])],
                    model.feature_importances_
                ))
            )
            
            self.dimension_mappers[dim_name] = mapping
            
            logger.info(f"Trained {dim_name} mapper: R² = {r2:.3f}, MSE = {mse:.3f}")
    
    def map_embedding_to_coordinates(self, embedding: np.ndarray) -> Dict[str, float]:
        """Map a semantic embedding to interpretable coordinates"""
        coordinates = {}
        
        for dim_name, mapper in self.dimension_mappers.items():
            try:
                coord_value = mapper.embedding_to_coord_model.predict([embedding])[0]
                # Ensure coordinate is in valid range [0, 1]
                coord_value = max(0.0, min(1.0, coord_value))
                coordinates[dim_name] = round(coord_value, 3)
            except Exception as e:
                logger.error(f"Error mapping {dim_name}: {e}")
                coordinates[dim_name] = 0.5  # Default fallback
        
        return coordinates
    
    def get_mapping_quality(self) -> Dict[str, Any]:
        """Get quality metrics for coordinate mappings"""
        quality_info = {}
        
        for dim_name, mapper in self.dimension_mappers.items():
            quality_info[dim_name] = {
                'validation_score': mapper.validation_score,
                'model_type': mapper.model_type,
                'training_examples': len(mapper.training_examples),
                'top_features': self._get_top_features(mapper.feature_importance) if mapper.feature_importance else None
            }
        
        return quality_info
    
    def _get_top_features(self, feature_importance: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top k most important features"""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]


class SemanticCoordinateEngine:
    """
    Semantic Coordinate Engine that replaces keyword-based heuristics
    with proper semantic understanding and learned coordinate mapping.
    
    This addresses the core flaw identified in the feedback.
    """
    
    def __init__(self, embedding_backend: str = 'auto'):
        self.embedding_backend = embedding_backend
        self.embedder = self._initialize_embedder()
        self.coordinate_mapper = CoordinateMapper()
        self.coordinate_cache = {}
        
        # Initialize with some default training data
        self._initialize_default_training_data()
        
        logger.info(f"SemanticCoordinateEngine initialized with {type(self.embedder).__name__}")
    
    def _initialize_embedder(self) -> SemanticEmbedder:
        """Initialize the best available semantic embedder"""
        
        if self.embedding_backend == 'sentence_transformers' and EMBEDDING_BACKENDS_AVAILABLE['sentence_transformers']:
            return SentenceTransformerEmbedder()
        elif self.embedding_backend == 'transformers' and EMBEDDING_BACKENDS_AVAILABLE['transformers']:
            return TransformerEmbedder()
        elif self.embedding_backend == 'auto':
            # Auto-select best available
            if EMBEDDING_BACKENDS_AVAILABLE['sentence_transformers']:
                return SentenceTransformerEmbedder()
            elif EMBEDDING_BACKENDS_AVAILABLE['transformers']:
                return TransformerEmbedder()
            elif EMBEDDING_BACKENDS_AVAILABLE['sklearn']:
                return TfidfEmbedder()
            else:
                raise ImportError("No embedding backends available")
        else:
            raise ValueError(f"Unsupported embedding backend: {self.embedding_backend}")
    
    def _initialize_default_training_data(self):
        """Initialize with some default training examples"""
        
        # Domain training examples
        domain_examples = [
            ("Python programming tutorial", {'domain': 0.9, 'complexity': 0.3, 'task_type': 0.2}),
            ("Machine learning algorithms", {'domain': 0.9, 'complexity': 0.8, 'task_type': 0.7}),
            ("Business strategy analysis", {'domain': 0.3, 'complexity': 0.6, 'task_type': 0.8}),
            ("Creative writing techniques", {'domain': 0.1, 'complexity': 0.4, 'task_type': 0.3}),
            ("Data science methodology", {'domain': 0.8, 'complexity': 0.7, 'task_type': 0.7}),
            ("Marketing campaign planning", {'domain': 0.2, 'complexity': 0.5, 'task_type': 0.6}),
            ("Software architecture patterns", {'domain': 0.9, 'complexity': 0.9, 'task_type': 0.5}),
            ("Basic HTML tutorial", {'domain': 0.8, 'complexity': 0.2, 'task_type': 0.2}),
            ("Advanced quantum computing", {'domain': 0.9, 'complexity': 1.0, 'task_type': 0.8}),
            ("Introduction to economics", {'domain': 0.4, 'complexity': 0.3, 'task_type': 0.4}),
        ]
        
        for text, coordinates in domain_examples:
            self.coordinate_mapper.add_training_example(text, coordinates)
        
        # Train the mappers
        self.coordinate_mapper.train_coordinate_mappers(self.embedder)
    
    def text_to_coordinates(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Convert text to semantic coordinates using learned mappings.
        
        This replaces the flawed keyword-based approach.
        """
        # Check cache
        if use_cache:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.coordinate_cache:
                return self.coordinate_cache[text_hash]
        
        # Generate semantic embedding
        try:
            embedding = self.embedder.embed_text(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback to default coordinates
            return {
                'coordinates': {'domain': 0.5, 'complexity': 0.5, 'task_type': 0.5},
                'method': 'fallback',
                'error': str(e),
                'embedding_quality': 0.0
            }
        
        # Map embedding to coordinates
        coordinates = self.coordinate_mapper.map_embedding_to_coordinates(embedding)
        
        # Calculate embedding quality (based on confidence in mapping)
        embedding_quality = self._calculate_embedding_quality(embedding, coordinates)
        
        # Generate explanation
        explanation = self._generate_semantic_explanation(text, embedding, coordinates)
        
        result = {
            'coordinates': coordinates,
            'method': 'semantic_embedding',
            'embedding_dimension': len(embedding),
            'embedding_quality': embedding_quality,
            'explanation': explanation,
            'mapper_quality': self.coordinate_mapper.get_mapping_quality()
        }
        
        # Cache result
        if use_cache:
            self.coordinate_cache[text_hash] = result
        
        return result
    
    def _calculate_embedding_quality(self, embedding: np.ndarray, coordinates: Dict[str, float]) -> float:
        """Calculate quality/confidence of the embedding and coordinate mapping"""
        
        # Basic quality metrics
        embedding_norm = np.linalg.norm(embedding)
        embedding_sparsity = np.sum(np.abs(embedding) < 0.01) / len(embedding)
        
        # Coordinate consistency (how well coordinates align with each other)
        coord_values = list(coordinates.values())
        coord_variance = np.var(coord_values) if len(coord_values) > 1 else 0.0
        
        # Mapper quality (average validation score)
        mapper_qualities = [mapper.validation_score for mapper in self.coordinate_mapper.dimension_mappers.values()]
        avg_mapper_quality = np.mean(mapper_qualities) if mapper_qualities else 0.0
        
        # Combine metrics
        quality = (
            min(1.0, embedding_norm / 10.0) * 0.3 +  # Embedding strength
            (1.0 - embedding_sparsity) * 0.2 +       # Embedding density
            min(1.0, coord_variance * 2) * 0.2 +     # Coordinate diversity
            avg_mapper_quality * 0.3                  # Mapper reliability
        )
        
        return max(0.0, min(1.0, quality))
    
    def _generate_semantic_explanation(self, text: str, embedding: np.ndarray, 
                                     coordinates: Dict[str, float]) -> str:
        """Generate explanation for semantic coordinate assignment"""
        
        explanations = []
        
        # Analyze embedding characteristics
        embedding_strength = np.linalg.norm(embedding)
        dominant_features = np.argsort(np.abs(embedding))[-5:]  # Top 5 features
        
        explanations.append(f"Semantic embedding strength: {embedding_strength:.2f}")
        explanations.append(f"Dominant semantic features: {len(dominant_features)} identified")
        
        # Analyze coordinates
        for dim_name, coord_value in coordinates.items():
            if dim_name in self.coordinate_mapper.dimension_mappers:
                mapper = self.coordinate_mapper.dimension_mappers[dim_name]
                confidence = mapper.validation_score
                explanations.append(
                    f"{dim_name.title()}: {coord_value:.3f} (confidence: {confidence:.2f})"
                )
        
        return "Semantic analysis: " + "; ".join(explanations)
    
    def add_training_example(self, text: str, coordinates: Dict[str, float]):
        """Add a new training example and retrain mappers"""
        self.coordinate_mapper.add_training_example(text, coordinates)
        
        # Retrain if we have enough new examples
        total_examples = sum(len(examples) for examples in self.coordinate_mapper.training_data.values())
        if total_examples % 50 == 0:  # Retrain every 50 examples
            logger.info("Retraining coordinate mappers with new examples...")
            self.coordinate_mapper.train_coordinate_mappers(self.embedder)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the semantic coordinate system"""
        return {
            'embedder_type': type(self.embedder).__name__,
            'embedding_dimension': self.embedder.embedding_dimension,
            'available_backends': EMBEDDING_BACKENDS_AVAILABLE,
            'trained_dimensions': list(self.coordinate_mapper.dimension_mappers.keys()),
            'mapper_quality': self.coordinate_mapper.get_mapping_quality(),
            'cache_size': len(self.coordinate_cache),
            'total_training_examples': sum(
                len(examples) for examples in self.coordinate_mapper.training_data.values()
            )
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the semantic coordinate engine
    try:
        semantic_engine = SemanticCoordinateEngine()
        
        print("Semantic Coordinate Engine Demo")
        print("=" * 40)
        
        # Test various texts
        test_texts = [
            "Advanced Python machine learning optimization techniques",
            "Basic HTML tutorial for beginners",
            "Business strategy and market analysis",
            "Creative writing and storytelling methods",
            "Quantum computing algorithms and implementation"
        ]
        
        for text in test_texts:
            print(f"\nText: {text}")
            result = semantic_engine.text_to_coordinates(text)
            
            print(f"Coordinates: {result['coordinates']}")
            print(f"Method: {result['method']}")
            print(f"Quality: {result['embedding_quality']:.3f}")
            print(f"Explanation: {result['explanation']}")
        
        # Show system status
        status = semantic_engine.get_system_status()
        print(f"\nSystem Status:")
        print(f"  Embedder: {status['embedder_type']}")
        print(f"  Embedding dimension: {status['embedding_dimension']}")
        print(f"  Trained dimensions: {status['trained_dimensions']}")
        print(f"  Cache size: {status['cache_size']}")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        print("This may be due to missing dependencies. Install sentence-transformers or transformers for full functionality.")