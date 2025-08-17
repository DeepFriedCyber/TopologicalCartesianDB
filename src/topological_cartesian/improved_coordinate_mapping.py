#!/usr/bin/env python3
"""
Improved Coordinate Mapping System

Addresses the coordinate separation quality issues by:
1. Better training data collection and labeling
2. Supervised learning for coordinate mapping
3. Domain-specific coordinate spaces
4. Active learning for continuous improvement
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """Enhanced training example with metadata"""
    text: str
    coordinates: Dict[str, float]
    domain: str
    confidence: float
    source: str
    timestamp: float = field(default_factory=time.time)
    validated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'coordinates': self.coordinates,
            'domain': self.domain,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.timestamp,
            'validated': self.validated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        return cls(**data)


class DomainSpecificCoordinateMapper:
    """Domain-specific coordinate mapping with supervised learning"""
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.models = {}
        self.scalers = {}
        self.training_examples = []
        self.is_trained = False
        self.model_performance = {}
        
        # Model configurations for different coordinate dimensions
        self.model_configs = {
            'random_forest': {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'neural_network': {
                'class': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'max_iter': 500,
                    'random_state': 42
                }
            }
        }
    
    def add_training_examples(self, examples: List[TrainingExample]):
        """Add multiple training examples"""
        domain_examples = [ex for ex in examples if ex.domain == self.domain or self.domain == "general"]
        self.training_examples.extend(domain_examples)
        logger.info(f"Added {len(domain_examples)} training examples for domain '{self.domain}'")
    
    def prepare_training_data(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data from examples and embeddings"""
        if len(self.training_examples) != len(embeddings):
            raise ValueError(f"Mismatch: {len(self.training_examples)} examples vs {len(embeddings)} embeddings")
        
        X = embeddings
        y_dict = {}
        
        # Extract coordinate targets
        coordinate_names = set()
        for example in self.training_examples:
            coordinate_names.update(example.coordinates.keys())
        
        for coord_name in coordinate_names:
            y_values = []
            for example in self.training_examples:
                y_values.append(example.coordinates.get(coord_name, 0.5))  # Default to middle
            y_dict[coord_name] = np.array(y_values)
        
        return X, y_dict
    
    def train_models(self, embeddings: np.ndarray, model_type: str = 'random_forest'):
        """Train coordinate mapping models"""
        if len(self.training_examples) < 3:
            logger.debug(f"Insufficient training data: {len(self.training_examples)} examples (need at least 3)")
            return False
        
        X, y_dict = self.prepare_training_data(embeddings)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model_config = self.model_configs[model_type]
        
        for coord_name, y in y_dict.items():
            logger.info(f"Training model for coordinate '{coord_name}'...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = model_config['class'](**model_config['params'])
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            
            self.models[coord_name] = model
            self.scalers[coord_name] = scaler
            self.model_performance[coord_name] = {
                'mse': mse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"  {coord_name}: R² = {r2:.3f}, CV = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.is_trained = True
        return True
    
    def predict_coordinates(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict coordinates for new embeddings"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        coordinates = {}
        
        for coord_name, model in self.models.items():
            scaler = self.scalers[coord_name]
            X_scaled = scaler.transform(embeddings)
            predictions = model.predict(X_scaled)
            
            # Ensure predictions are in [0, 1] range
            predictions = np.clip(predictions, 0.0, 1.0)
            coordinates[coord_name] = predictions
        
        return coordinates
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.model_performance
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        model_data = {
            'domain': self.domain,
            'models': self.models,
            'scalers': self.scalers,
            'performance': self.model_performance,
            'training_examples_count': len(self.training_examples),
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.domain = model_data['domain']
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.model_performance = model_data['performance']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Models loaded from {filepath}")


class TrainingDataCollector:
    """Collect and manage training data for coordinate mapping"""
    
    def __init__(self):
        self.examples = []
        self.domain_stats = {}
    
    def create_domain_specific_examples(self) -> List[TrainingExample]:
        """Create comprehensive domain-specific training examples"""
        
        examples = []
        
        # Programming/Technical domain
        programming_examples = [
            ("Python machine learning tutorial with scikit-learn", 
             {'domain': 0.9, 'complexity': 0.7, 'task_type': 0.3}, "programming", 0.9),
            ("Basic HTML CSS tutorial for beginners", 
             {'domain': 0.8, 'complexity': 0.2, 'task_type': 0.1}, "programming", 0.9),
            ("Advanced neural network architecture design", 
             {'domain': 0.95, 'complexity': 0.95, 'task_type': 0.8}, "programming", 0.9),
            ("Simple JavaScript function examples", 
             {'domain': 0.8, 'complexity': 0.3, 'task_type': 0.2}, "programming", 0.9),
            ("Database optimization and indexing strategies", 
             {'domain': 0.85, 'complexity': 0.8, 'task_type': 0.7}, "programming", 0.9),
            ("Introduction to programming concepts", 
             {'domain': 0.7, 'complexity': 0.1, 'task_type': 0.1}, "programming", 0.9),
            ("Distributed systems architecture patterns", 
             {'domain': 0.9, 'complexity': 0.9, 'task_type': 0.8}, "programming", 0.9),
            ("Web development responsive design basics", 
             {'domain': 0.75, 'complexity': 0.4, 'task_type': 0.3}, "programming", 0.9),
        ]
        
        # Science domain
        science_examples = [
            ("Quantum mechanics wave function analysis", 
             {'domain': 0.95, 'complexity': 0.95, 'task_type': 0.9}, "science", 0.9),
            ("Basic chemistry periodic table introduction", 
             {'domain': 0.7, 'complexity': 0.2, 'task_type': 0.1}, "science", 0.9),
            ("Climate change environmental impact study", 
             {'domain': 0.8, 'complexity': 0.6, 'task_type': 0.7}, "science", 0.9),
            ("Elementary physics motion and forces", 
             {'domain': 0.7, 'complexity': 0.3, 'task_type': 0.2}, "science", 0.9),
            ("Advanced molecular biology research methods", 
             {'domain': 0.9, 'complexity': 0.8, 'task_type': 0.8}, "science", 0.9),
            ("Introduction to scientific method", 
             {'domain': 0.6, 'complexity': 0.2, 'task_type': 0.1}, "science", 0.9),
            ("Astrophysics black hole theoretical models", 
             {'domain': 0.95, 'complexity': 0.9, 'task_type': 0.9}, "science", 0.9),
            ("High school biology cell structure", 
             {'domain': 0.65, 'complexity': 0.3, 'task_type': 0.2}, "science", 0.9),
        ]
        
        # Business domain
        business_examples = [
            ("Strategic business planning and market analysis", 
             {'domain': 0.7, 'complexity': 0.7, 'task_type': 0.8}, "business", 0.9),
            ("Basic accounting principles for small business", 
             {'domain': 0.6, 'complexity': 0.3, 'task_type': 0.4}, "business", 0.9),
            ("Advanced financial derivatives trading strategies", 
             {'domain': 0.8, 'complexity': 0.9, 'task_type': 0.9}, "business", 0.9),
            ("Customer service best practices guide", 
             {'domain': 0.5, 'complexity': 0.2, 'task_type': 0.3}, "business", 0.9),
            ("Supply chain optimization methodologies", 
             {'domain': 0.75, 'complexity': 0.8, 'task_type': 0.7}, "business", 0.9),
            ("Introduction to entrepreneurship basics", 
             {'domain': 0.5, 'complexity': 0.2, 'task_type': 0.2}, "business", 0.9),
            ("Corporate governance and compliance frameworks", 
             {'domain': 0.7, 'complexity': 0.6, 'task_type': 0.8}, "business", 0.9),
            ("Marketing campaign performance metrics", 
             {'domain': 0.65, 'complexity': 0.5, 'task_type': 0.6}, "business", 0.9),
        ]
        
        # Creative domain
        creative_examples = [
            ("Advanced digital art composition techniques", 
             {'domain': 0.3, 'complexity': 0.7, 'task_type': 0.2}, "creative", 0.9),
            ("Basic drawing fundamentals for beginners", 
             {'domain': 0.2, 'complexity': 0.1, 'task_type': 0.1}, "creative", 0.9),
            ("Creative writing narrative structure analysis", 
             {'domain': 0.25, 'complexity': 0.6, 'task_type': 0.3}, "creative", 0.9),
            ("Photography lighting and exposure basics", 
             {'domain': 0.3, 'complexity': 0.3, 'task_type': 0.2}, "creative", 0.9),
            ("Music theory harmony and composition", 
             {'domain': 0.35, 'complexity': 0.8, 'task_type': 0.4}, "creative", 0.9),
            ("Simple craft projects for children", 
             {'domain': 0.1, 'complexity': 0.1, 'task_type': 0.1}, "creative", 0.9),
            ("Film production cinematography techniques", 
             {'domain': 0.4, 'complexity': 0.7, 'task_type': 0.5}, "creative", 0.9),
            ("Graphic design typography principles", 
             {'domain': 0.35, 'complexity': 0.5, 'task_type': 0.3}, "creative", 0.9),
        ]
        
        # Convert to TrainingExample objects
        all_examples = programming_examples + science_examples + business_examples + creative_examples
        
        for text, coords, domain, confidence in all_examples:
            example = TrainingExample(
                text=text,
                coordinates=coords,
                domain=domain,
                confidence=confidence,
                source="manual_curation",
                validated=True
            )
            examples.append(example)
        
        return examples
    
    def create_complexity_gradient_examples(self) -> List[TrainingExample]:
        """Create examples with clear complexity gradients"""
        
        examples = []
        
        # Programming complexity gradient
        programming_complexity = [
            ("Hello world program", {'domain': 0.8, 'complexity': 0.05, 'task_type': 0.1}),
            ("Variables and basic operations", {'domain': 0.8, 'complexity': 0.1, 'task_type': 0.1}),
            ("Functions and control flow", {'domain': 0.8, 'complexity': 0.2, 'task_type': 0.2}),
            ("Object-oriented programming basics", {'domain': 0.8, 'complexity': 0.3, 'task_type': 0.3}),
            ("Data structures and algorithms", {'domain': 0.8, 'complexity': 0.5, 'task_type': 0.4}),
            ("Design patterns implementation", {'domain': 0.8, 'complexity': 0.7, 'task_type': 0.6}),
            ("Concurrent programming and threading", {'domain': 0.8, 'complexity': 0.8, 'task_type': 0.7}),
            ("Distributed systems architecture", {'domain': 0.8, 'complexity': 0.95, 'task_type': 0.9}),
        ]
        
        # Science complexity gradient
        science_complexity = [
            ("Basic scientific observation", {'domain': 0.7, 'complexity': 0.05, 'task_type': 0.1}),
            ("Simple experiments and measurements", {'domain': 0.7, 'complexity': 0.1, 'task_type': 0.2}),
            ("Data collection and basic analysis", {'domain': 0.7, 'complexity': 0.2, 'task_type': 0.3}),
            ("Hypothesis testing and statistics", {'domain': 0.7, 'complexity': 0.4, 'task_type': 0.5}),
            ("Advanced statistical modeling", {'domain': 0.7, 'complexity': 0.6, 'task_type': 0.7}),
            ("Theoretical framework development", {'domain': 0.7, 'complexity': 0.8, 'task_type': 0.8}),
            ("Complex mathematical proofs", {'domain': 0.7, 'complexity': 0.95, 'task_type': 0.9}),
        ]
        
        # Convert to TrainingExample objects
        all_complexity_examples = programming_complexity + science_complexity
        
        for text, coords in all_complexity_examples:
            example = TrainingExample(
                text=text,
                coordinates=coords,
                domain="general",
                confidence=0.95,
                source="complexity_gradient",
                validated=True
            )
            examples.append(example)
        
        return examples
    
    def create_task_type_examples(self) -> List[TrainingExample]:
        """Create examples with clear task type distinctions"""
        
        examples = []
        
        # Learning/Educational (low task_type)
        learning_examples = [
            ("Introduction to basic concepts", {'domain': 0.5, 'complexity': 0.3, 'task_type': 0.05}),
            ("Tutorial with step-by-step instructions", {'domain': 0.5, 'complexity': 0.4, 'task_type': 0.1}),
            ("Educational overview and explanation", {'domain': 0.5, 'complexity': 0.3, 'task_type': 0.1}),
            ("Beginner's guide to fundamentals", {'domain': 0.5, 'complexity': 0.2, 'task_type': 0.05}),
        ]
        
        # Analysis/Research (medium task_type)
        analysis_examples = [
            ("Comparative analysis of different approaches", {'domain': 0.5, 'complexity': 0.6, 'task_type': 0.5}),
            ("Research methodology and findings", {'domain': 0.5, 'complexity': 0.7, 'task_type': 0.6}),
            ("Data analysis and interpretation", {'domain': 0.5, 'complexity': 0.6, 'task_type': 0.5}),
            ("Performance evaluation and benchmarking", {'domain': 0.5, 'complexity': 0.7, 'task_type': 0.6}),
        ]
        
        # Implementation/Production (high task_type)
        implementation_examples = [
            ("Production deployment and scaling", {'domain': 0.5, 'complexity': 0.8, 'task_type': 0.9}),
            ("System architecture and implementation", {'domain': 0.5, 'complexity': 0.8, 'task_type': 0.85}),
            ("Optimization and performance tuning", {'domain': 0.5, 'complexity': 0.7, 'task_type': 0.8}),
            ("Enterprise solution development", {'domain': 0.5, 'complexity': 0.8, 'task_type': 0.9}),
        ]
        
        # Convert to TrainingExample objects
        all_task_examples = learning_examples + analysis_examples + implementation_examples
        
        for text, coords in all_task_examples:
            example = TrainingExample(
                text=text,
                coordinates=coords,
                domain="general",
                confidence=0.9,
                source="task_type_gradient",
                validated=True
            )
            examples.append(example)
        
        return examples
    
    def collect_comprehensive_training_data(self) -> List[TrainingExample]:
        """Collect comprehensive training data"""
        
        all_examples = []
        
        # Domain-specific examples
        all_examples.extend(self.create_domain_specific_examples())
        
        # Complexity gradient examples
        all_examples.extend(self.create_complexity_gradient_examples())
        
        # Task type examples
        all_examples.extend(self.create_task_type_examples())
        
        # Update statistics
        self.examples = all_examples
        self.update_domain_stats()
        
        logger.info(f"Collected {len(all_examples)} comprehensive training examples")
        logger.info(f"Domain distribution: {self.domain_stats}")
        
        return all_examples
    
    def update_domain_stats(self):
        """Update domain statistics"""
        self.domain_stats = {}
        for example in self.examples:
            domain = example.domain
            if domain not in self.domain_stats:
                self.domain_stats[domain] = 0
            self.domain_stats[domain] += 1
    
    def save_training_data(self, filepath: str):
        """Save training data to file"""
        data = {
            'examples': [ex.to_dict() for ex in self.examples],
            'domain_stats': self.domain_stats,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Training data saved to {filepath}")
    
    def load_training_data(self, filepath: str):
        """Load training data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.examples = [TrainingExample.from_dict(ex_data) for ex_data in data['examples']]
        self.domain_stats = data['domain_stats']
        
        logger.info(f"Training data loaded from {filepath}")


class ImprovedCoordinateEngine:
    """Enhanced coordinate engine with supervised learning"""
    
    def __init__(self):
        self.base_embedder = None
        self.domain_mappers = {}
        self.training_collector = TrainingDataCollector()
        self.is_trained = False
        
        # Initialize base embedder
        self._initialize_embedder()
        
        # Collect training data
        self.training_examples = self.training_collector.collect_comprehensive_training_data()
    
    def _initialize_embedder(self):
        """Initialize the base embedding system"""
        try:
            from sentence_transformers import SentenceTransformer
            self.base_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized SentenceTransformer embedder")
        except ImportError:
            # Fallback to TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.base_embedder = TfidfVectorizer(max_features=384, stop_words='english')
            logger.info("Initialized TF-IDF embedder (fallback)")
    
    def train_domain_mappers(self, domains: List[str] = None):
        """Train coordinate mappers for specific domains"""
        
        if domains is None:
            domains = list(self.training_collector.domain_stats.keys())
        
        # Generate embeddings for all training examples
        texts = [ex.text for ex in self.training_examples]
        
        if hasattr(self.base_embedder, 'encode'):
            # SentenceTransformer
            embeddings = self.base_embedder.encode(texts)
        else:
            # TF-IDF
            embeddings = self.base_embedder.fit_transform(texts).toarray()
        
        # Train mappers for each domain
        for domain in domains:
            logger.debug(f"Training coordinate mapper for domain: {domain}")
            
            mapper = DomainSpecificCoordinateMapper(domain)
            
            # Filter examples for this domain
            domain_examples = [ex for ex in self.training_examples 
                             if ex.domain == domain or domain == "general"]
            
            if len(domain_examples) < 3:
                logger.debug(f"Insufficient examples for domain {domain}: {len(domain_examples)}")
                continue
            
            # Get corresponding embeddings
            domain_indices = [i for i, ex in enumerate(self.training_examples) 
                            if ex.domain == domain or domain == "general"]
            domain_embeddings = embeddings[domain_indices]
            
            mapper.add_training_examples(domain_examples)
            
            # Try different model types and pick the best
            best_model = None
            best_performance = -float('inf')
            
            for model_type in ['random_forest', 'gradient_boosting', 'neural_network']:
                try:
                    mapper_copy = DomainSpecificCoordinateMapper(domain)
                    mapper_copy.add_training_examples(domain_examples)
                    
                    success = mapper_copy.train_models(domain_embeddings, model_type)
                    if success:
                        performance = mapper_copy.get_model_performance()
                        avg_r2 = np.mean([perf['r2'] for perf in performance.values()])
                        
                        if avg_r2 > best_performance:
                            best_performance = avg_r2
                            best_model = mapper_copy
                            logger.info(f"  {model_type}: R² = {avg_r2:.3f} (best so far)")
                        else:
                            logger.info(f"  {model_type}: R² = {avg_r2:.3f}")
                
                except Exception as e:
                    logger.warning(f"  {model_type} failed: {e}")
            
            if best_model:
                self.domain_mappers[domain] = best_model
                logger.info(f"  Best model for {domain}: R² = {best_performance:.3f}")
            else:
                logger.debug(f"  No successful model for domain {domain}")
        
        self.is_trained = len(self.domain_mappers) > 0
        return self.is_trained
    
    def predict_coordinates(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """Predict coordinates for text using trained models"""
        
        if not self.is_trained:
            logger.warning("Models not trained, using fallback")
            return self._fallback_coordinates(text)
        
        # Get embedding
        if hasattr(self.base_embedder, 'encode'):
            embedding = self.base_embedder.encode([text])
        else:
            embedding = self.base_embedder.transform([text]).toarray()
        
        # Use domain-specific mapper if available
        mapper = self.domain_mappers.get(domain, self.domain_mappers.get("general"))
        
        if mapper is None:
            logger.warning(f"No mapper for domain {domain}, using fallback")
            return self._fallback_coordinates(text)
        
        try:
            coordinates = mapper.predict_coordinates(embedding)
            
            # Convert to single values
            coord_dict = {}
            for coord_name, values in coordinates.items():
                coord_dict[coord_name] = float(values[0])
            
            return {
                'coordinates': coord_dict,
                'method': 'supervised_learning',
                'domain': domain,
                'model_performance': mapper.get_model_performance(),
                'confidence': self._calculate_confidence(coord_dict, mapper)
            }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback_coordinates(text)
    
    def _fallback_coordinates(self, text: str) -> Dict[str, Any]:
        """Fallback coordinate generation"""
        # Simple heuristic-based fallback
        words = text.lower().split()
        
        # Domain estimation
        tech_words = {'python', 'programming', 'algorithm', 'code', 'software', 'computer'}
        science_words = {'research', 'study', 'analysis', 'theory', 'experiment', 'data'}
        business_words = {'business', 'market', 'strategy', 'management', 'finance', 'customer'}
        
        tech_score = sum(1 for word in words if word in tech_words) / max(len(words), 1)
        science_score = sum(1 for word in words if word in science_words) / max(len(words), 1)
        business_score = sum(1 for word in words if word in business_words) / max(len(words), 1)
        
        domain = max(tech_score, science_score, business_score, 0.3)
        
        # Complexity estimation
        complex_words = {'advanced', 'complex', 'sophisticated', 'intricate', 'comprehensive'}
        simple_words = {'basic', 'simple', 'introduction', 'beginner', 'elementary'}
        
        complex_score = sum(1 for word in words if word in complex_words)
        simple_score = sum(1 for word in words if word in simple_words)
        
        complexity = 0.5 + (complex_score - simple_score) * 0.1
        complexity = max(0.0, min(1.0, complexity))
        
        # Task type estimation
        task_words = {'implementation', 'production', 'deployment', 'system', 'architecture'}
        learn_words = {'tutorial', 'guide', 'introduction', 'learn', 'understand'}
        
        task_score = sum(1 for word in words if word in task_words)
        learn_score = sum(1 for word in words if word in learn_words)
        
        task_type = 0.5 + (task_score - learn_score) * 0.1
        task_type = max(0.0, min(1.0, task_type))
        
        return {
            'coordinates': {
                'domain': domain,
                'complexity': complexity,
                'task_type': task_type
            },
            'method': 'heuristic_fallback',
            'domain': 'general',
            'confidence': 0.3
        }
    
    def _calculate_confidence(self, coordinates: Dict[str, float], mapper: DomainSpecificCoordinateMapper) -> float:
        """Calculate prediction confidence based on model performance"""
        performance = mapper.get_model_performance()
        
        if not performance:
            return 0.5
        
        # Average R² score across all coordinates
        r2_scores = [perf['r2'] for perf in performance.values()]
        avg_r2 = np.mean(r2_scores)
        
        # Convert R² to confidence (R² can be negative for very bad models)
        confidence = max(0.0, min(1.0, avg_r2))
        
        return confidence
    
    def save_models(self, directory: str):
        """Save all trained models"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for domain, mapper in self.domain_mappers.items():
            filepath = Path(directory) / f"{domain}_mapper.pkl"
            mapper.save_models(str(filepath))
        
        # Save training data
        training_filepath = Path(directory) / "training_data.json"
        self.training_collector.save_training_data(str(training_filepath))
        
        logger.info(f"All models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load all trained models"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Model directory does not exist: {directory}")
            return False
        
        # Load training data
        training_filepath = directory_path / "training_data.json"
        if training_filepath.exists():
            self.training_collector.load_training_data(str(training_filepath))
        
        # Load mappers
        for model_file in directory_path.glob("*_mapper.pkl"):
            domain = model_file.stem.replace("_mapper", "")
            mapper = DomainSpecificCoordinateMapper(domain)
            mapper.load_models(str(model_file))
            self.domain_mappers[domain] = mapper
        
        self.is_trained = len(self.domain_mappers) > 0
        logger.info(f"Loaded {len(self.domain_mappers)} domain mappers")
        
        return self.is_trained


if __name__ == "__main__":
    # Demonstration of improved coordinate mapping
    print("Improved Coordinate Mapping System")
    print("=" * 40)
    
    # Initialize and train
    engine = ImprovedCoordinateEngine()
    
    print(f"Collected {len(engine.training_examples)} training examples")
    print(f"Domain distribution: {engine.training_collector.domain_stats}")
    
    # Train models
    print("\nTraining domain-specific models...")
    success = engine.train_domain_mappers()
    
    if success:
        print(f"✅ Successfully trained {len(engine.domain_mappers)} domain mappers")
        
        # Test predictions
        test_texts = [
            ("Advanced Python machine learning algorithms", "programming"),
            ("Basic HTML tutorial for beginners", "programming"),
            ("Quantum mechanics theoretical framework", "science"),
            ("Business strategy market analysis", "business"),
            ("Creative writing narrative techniques", "creative"),
        ]
        
        print("\nTesting predictions:")
        print("-" * 30)
        
        for text, domain in test_texts:
            result = engine.predict_coordinates(text, domain)
            coords = result['coordinates']
            
            print(f"\nText: {text}")
            print(f"Domain: {domain}")
            print(f"Coordinates: {coords}")
            print(f"Method: {result['method']}")
            print(f"Confidence: {result.get('confidence', 0):.3f}")
            
            if 'model_performance' in result:
                perf = result['model_performance']
                avg_r2 = np.mean([p['r2'] for p in perf.values()])
                print(f"Model R²: {avg_r2:.3f}")
        
        # Save models
        engine.save_models("trained_models")
        print(f"\n✅ Models saved to 'trained_models' directory")
        
    else:
        print("❌ Training failed")