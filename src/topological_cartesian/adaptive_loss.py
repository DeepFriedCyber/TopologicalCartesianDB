#!/usr/bin/env python3
"""
Adaptive Query Loss - Dynamic Loss Function Optimization

Implements adaptive loss functions that change over time when performance
stagnates, enabling continuous improvement and escape from local minima
in query processing optimization.

Based on DNN optimization research from:
https://www.datasciencecentral.com/how-to-build-and-optimize-high-performance-deep-neural-networks-from-scratch/
"""

import numpy as np
import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import json

logger = logging.getLogger(__name__)


class LossFunctionType(Enum):
    """Types of loss functions for query optimization"""
    MEAN_SQUARED_ERROR = "mse"
    MEAN_ABSOLUTE_ERROR = "mae"
    HUBER_LOSS = "huber"
    FOCAL_LOSS = "focal"
    CONTRASTIVE_LOSS = "contrastive"
    TRIPLET_LOSS = "triplet"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


@dataclass
class LossConfiguration:
    """Configuration for a specific loss function"""
    loss_type: LossFunctionType
    parameters: Dict[str, float]
    weight: float = 1.0
    active: bool = True
    performance_threshold: float = 0.01
    stagnation_patience: int = 5


@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for query processing"""
    accuracy: float
    processing_time: float
    coherence: float
    resource_efficiency: float
    user_satisfaction: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LossEvaluationResult:
    """Result of loss function evaluation"""
    loss_value: float
    gradient: np.ndarray
    loss_type: LossFunctionType
    parameters_used: Dict[str, float]
    computation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationEvent:
    """Records when and why loss function adaptation occurred"""
    timestamp: float
    previous_loss_type: LossFunctionType
    new_loss_type: LossFunctionType
    trigger_reason: str
    performance_before: float
    performance_after: Optional[float] = None
    adaptation_success: Optional[bool] = None


class LossFunctionLibrary:
    """Library of different loss functions for query optimization"""
    
    @staticmethod
    def mean_squared_error(predicted: np.ndarray, actual: np.ndarray, 
                          parameters: Dict[str, float]) -> Tuple[float, np.ndarray]:
        """Mean Squared Error loss function"""
        diff = predicted - actual
        loss = np.mean(diff ** 2)
        gradient = 2 * diff / len(diff)
        return loss, gradient
    
    @staticmethod
    def mean_absolute_error(predicted: np.ndarray, actual: np.ndarray, 
                           parameters: Dict[str, float]) -> Tuple[float, np.ndarray]:
        """Mean Absolute Error loss function"""
        diff = predicted - actual
        loss = np.mean(np.abs(diff))
        gradient = np.sign(diff) / len(diff)
        return loss, gradient
    
    @staticmethod
    def huber_loss(predicted: np.ndarray, actual: np.ndarray, 
                   parameters: Dict[str, float]) -> Tuple[float, np.ndarray]:
        """Huber loss function (robust to outliers)"""
        delta = parameters.get('delta', 1.0)
        diff = predicted - actual
        abs_diff = np.abs(diff)
        
        # Huber loss: quadratic for small errors, linear for large errors
        loss_values = np.where(abs_diff <= delta,
                              0.5 * diff ** 2,
                              delta * (abs_diff - 0.5 * delta))
        loss = np.mean(loss_values)
        
        # Gradient
        gradient = np.where(abs_diff <= delta,
                           diff,
                           delta * np.sign(diff))
        gradient = gradient / len(gradient)
        
        return loss, gradient
    
    @staticmethod
    def focal_loss(predicted: np.ndarray, actual: np.ndarray, 
                   parameters: Dict[str, float]) -> Tuple[float, np.ndarray]:
        """Focal loss function (focuses on hard examples)"""
        alpha = parameters.get('alpha', 1.0)
        gamma = parameters.get('gamma', 2.0)
        epsilon = 1e-7
        
        # Ensure predicted values are in valid range
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        
        # Focal loss computation
        pt = np.where(actual == 1, predicted, 1 - predicted)
        focal_weight = alpha * (1 - pt) ** gamma
        
        cross_entropy = -np.log(pt)
        loss = np.mean(focal_weight * cross_entropy)
        
        # Simplified gradient (approximation)
        gradient = focal_weight * (predicted - actual) / len(predicted)
        
        return loss, gradient
    
    @staticmethod
    def contrastive_loss(predicted: np.ndarray, actual: np.ndarray, 
                        parameters: Dict[str, float]) -> Tuple[float, np.ndarray]:
        """Contrastive loss for similarity learning"""
        margin = parameters.get('margin', 1.0)
        
        # Assume actual contains similarity labels (1 for similar, 0 for dissimilar)
        distance = np.linalg.norm(predicted - actual, axis=-1)
        
        # Contrastive loss
        loss_similar = actual * distance ** 2
        loss_dissimilar = (1 - actual) * np.maximum(0, margin - distance) ** 2
        loss = np.mean(loss_similar + loss_dissimilar)
        
        # Simplified gradient
        gradient = 2 * (predicted - actual) / len(predicted)
        
        return loss, gradient
    
    @staticmethod
    def adaptive_hybrid_loss(predicted: np.ndarray, actual: np.ndarray, 
                           parameters: Dict[str, float]) -> Tuple[float, np.ndarray]:
        """Adaptive hybrid loss combining multiple loss functions"""
        mse_weight = parameters.get('mse_weight', 0.4)
        mae_weight = parameters.get('mae_weight', 0.3)
        huber_weight = parameters.get('huber_weight', 0.3)
        
        # Compute individual losses
        mse_loss, mse_grad = LossFunctionLibrary.mean_squared_error(predicted, actual, parameters)
        mae_loss, mae_grad = LossFunctionLibrary.mean_absolute_error(predicted, actual, parameters)
        huber_loss, huber_grad = LossFunctionLibrary.huber_loss(predicted, actual, parameters)
        
        # Combine losses
        total_loss = (mse_weight * mse_loss + 
                     mae_weight * mae_loss + 
                     huber_weight * huber_loss)
        
        total_gradient = (mse_weight * mse_grad + 
                         mae_weight * mae_grad + 
                         huber_weight * huber_grad)
        
        return total_loss, total_gradient


class PerformanceStagnationDetector:
    """Detects when query processing performance has stagnated"""
    
    def __init__(self, patience: int = 5, threshold: float = 0.01):
        self.patience = patience
        self.threshold = threshold
        self.performance_history = deque(maxlen=patience * 2)
        self.stagnation_counter = 0
        self.last_improvement_time = time.time()
    
    def update_performance(self, metrics: QueryPerformanceMetrics) -> bool:
        """Update performance and check for stagnation"""
        
        # Calculate composite performance score
        composite_score = (
            0.3 * metrics.accuracy +
            0.2 * (1.0 - min(1.0, metrics.processing_time / 10.0)) +  # Normalize time
            0.2 * metrics.coherence +
            0.15 * metrics.resource_efficiency +
            0.15 * metrics.user_satisfaction
        )
        
        self.performance_history.append({
            'score': composite_score,
            'timestamp': metrics.timestamp,
            'metrics': metrics
        })
        
        # Check for improvement
        if len(self.performance_history) >= 2:
            recent_scores = [p['score'] for p in list(self.performance_history)[-self.patience:]]
            
            if len(recent_scores) >= self.patience:
                # Check if recent performance is improving
                recent_avg = np.mean(recent_scores[-3:]) if len(recent_scores) >= 3 else recent_scores[-1]
                older_avg = np.mean(recent_scores[:-3]) if len(recent_scores) >= 6 else recent_scores[0]
                
                improvement = recent_avg - older_avg
                
                if improvement > self.threshold:
                    # Performance is improving
                    self.stagnation_counter = 0
                    self.last_improvement_time = time.time()
                    return False
                else:
                    # Performance is stagnating
                    self.stagnation_counter += 1
                    return self.stagnation_counter >= self.patience
        
        return False
    
    def get_stagnation_info(self) -> Dict[str, Any]:
        """Get information about current stagnation state"""
        
        recent_scores = [p['score'] for p in self.performance_history]
        
        return {
            'is_stagnating': self.stagnation_counter >= self.patience,
            'stagnation_counter': self.stagnation_counter,
            'time_since_improvement': time.time() - self.last_improvement_time,
            'recent_performance_trend': recent_scores[-5:] if len(recent_scores) >= 5 else recent_scores,
            'avg_recent_performance': np.mean(recent_scores[-5:]) if len(recent_scores) >= 5 else 0.0
        }


class AdaptiveLossEngine:
    """Core engine for adaptive loss function management"""
    
    def __init__(self):
        self.current_loss_config = LossConfiguration(
            loss_type=LossFunctionType.MEAN_SQUARED_ERROR,
            parameters={'delta': 1.0},
            weight=1.0
        )
        
        self.available_loss_functions = {
            LossFunctionType.MEAN_SQUARED_ERROR: LossFunctionLibrary.mean_squared_error,
            LossFunctionType.MEAN_ABSOLUTE_ERROR: LossFunctionLibrary.mean_absolute_error,
            LossFunctionType.HUBER_LOSS: LossFunctionLibrary.huber_loss,
            LossFunctionType.FOCAL_LOSS: LossFunctionLibrary.focal_loss,
            LossFunctionType.CONTRASTIVE_LOSS: LossFunctionLibrary.contrastive_loss,
            LossFunctionType.ADAPTIVE_HYBRID: LossFunctionLibrary.adaptive_hybrid_loss
        }
        
        self.stagnation_detector = PerformanceStagnationDetector()
        self.adaptation_history = []
        self.performance_history = []
        self.lock = threading.Lock()
        
        # Loss function performance tracking
        self.loss_function_performance = defaultdict(list)
        
    def compute_loss(self, predicted: np.ndarray, actual: np.ndarray, 
                    epoch: int, query_context: Dict[str, Any]) -> LossEvaluationResult:
        """Compute loss using current adaptive loss function"""
        
        start_time = time.time()
        
        with self.lock:
            # Get current loss function
            loss_func = self.available_loss_functions[self.current_loss_config.loss_type]
            
            # Compute loss and gradient
            loss_value, gradient = loss_func(predicted, actual, self.current_loss_config.parameters)
            
            computation_time = time.time() - start_time
            
            # Create result
            result = LossEvaluationResult(
                loss_value=loss_value,
                gradient=gradient,
                loss_type=self.current_loss_config.loss_type,
                parameters_used=self.current_loss_config.parameters.copy(),
                computation_time=computation_time,
                metadata={
                    'epoch': epoch,
                    'query_context': query_context,
                    'predicted_shape': predicted.shape,
                    'actual_shape': actual.shape
                }
            )
            
            # Track performance of current loss function
            self.loss_function_performance[self.current_loss_config.loss_type].append({
                'loss_value': loss_value,
                'epoch': epoch,
                'computation_time': computation_time
            })
            
            return result
    
    def update_performance_and_adapt(self, metrics: QueryPerformanceMetrics) -> Optional[AdaptationEvent]:
        """Update performance metrics and adapt loss function if needed"""
        
        with self.lock:
            self.performance_history.append(metrics)
            
            # Check for stagnation
            is_stagnating = self.stagnation_detector.update_performance(metrics)
            
            if is_stagnating:
                # Trigger adaptation
                adaptation_event = self._adapt_loss_function(metrics)
                if adaptation_event:
                    self.adaptation_history.append(adaptation_event)
                    logger.info(f"🔄 Loss function adapted: {adaptation_event.previous_loss_type.value} -> "
                               f"{adaptation_event.new_loss_type.value} (reason: {adaptation_event.trigger_reason})")
                    return adaptation_event
            
            return None
    
    def _adapt_loss_function(self, current_metrics: QueryPerformanceMetrics) -> Optional[AdaptationEvent]:
        """Adapt the loss function based on current performance"""
        
        previous_loss_type = self.current_loss_config.loss_type
        
        # Analyze current performance to choose new loss function
        new_loss_type = self._select_optimal_loss_function(current_metrics)
        
        if new_loss_type == previous_loss_type:
            return None  # No change needed
        
        # Create adaptation event
        adaptation_event = AdaptationEvent(
            timestamp=time.time(),
            previous_loss_type=previous_loss_type,
            new_loss_type=new_loss_type,
            trigger_reason=self._get_adaptation_reason(current_metrics),
            performance_before=self._calculate_composite_performance(current_metrics)
        )
        
        # Update loss configuration
        self.current_loss_config = self._create_loss_config(new_loss_type, current_metrics)
        
        # Reset stagnation detector
        self.stagnation_detector = PerformanceStagnationDetector()
        
        return adaptation_event
    
    def _select_optimal_loss_function(self, metrics: QueryPerformanceMetrics) -> LossFunctionType:
        """Select optimal loss function based on current performance characteristics"""
        
        # Analyze performance characteristics
        accuracy_low = metrics.accuracy < 0.7
        processing_slow = metrics.processing_time > 5.0
        coherence_low = metrics.coherence < 0.6
        resource_inefficient = metrics.resource_efficiency < 0.6
        
        # Decision logic for loss function selection
        if accuracy_low and coherence_low:
            # Focus on hard examples and coherence
            return LossFunctionType.FOCAL_LOSS
        elif processing_slow and resource_inefficient:
            # Need robust optimization
            return LossFunctionType.HUBER_LOSS
        elif metrics.accuracy > 0.8 and metrics.coherence > 0.7:
            # Fine-tuning phase - use contrastive loss
            return LossFunctionType.CONTRASTIVE_LOSS
        elif self._has_outliers_in_recent_performance():
            # Robust to outliers
            return LossFunctionType.HUBER_LOSS
        else:
            # Balanced approach
            return LossFunctionType.ADAPTIVE_HYBRID
    
    def _create_loss_config(self, loss_type: LossFunctionType, 
                           metrics: QueryPerformanceMetrics) -> LossConfiguration:
        """Create loss configuration based on loss type and current metrics"""
        
        parameters = {}
        
        if loss_type == LossFunctionType.HUBER_LOSS:
            # Adjust delta based on performance variance
            delta = 1.0 if metrics.accuracy > 0.7 else 0.5
            parameters['delta'] = delta
            
        elif loss_type == LossFunctionType.FOCAL_LOSS:
            # Adjust focus based on accuracy
            alpha = 1.0
            gamma = 3.0 if metrics.accuracy < 0.6 else 2.0
            parameters.update({'alpha': alpha, 'gamma': gamma})
            
        elif loss_type == LossFunctionType.CONTRASTIVE_LOSS:
            # Adjust margin based on coherence
            margin = 2.0 if metrics.coherence < 0.7 else 1.0
            parameters['margin'] = margin
            
        elif loss_type == LossFunctionType.ADAPTIVE_HYBRID:
            # Balance weights based on current performance
            if metrics.accuracy < 0.7:
                parameters.update({'mse_weight': 0.5, 'mae_weight': 0.3, 'huber_weight': 0.2})
            else:
                parameters.update({'mse_weight': 0.3, 'mae_weight': 0.3, 'huber_weight': 0.4})
        
        return LossConfiguration(
            loss_type=loss_type,
            parameters=parameters,
            weight=1.0,
            performance_threshold=0.01,
            stagnation_patience=5
        )
    
    def _get_adaptation_reason(self, metrics: QueryPerformanceMetrics) -> str:
        """Get human-readable reason for adaptation"""
        
        stagnation_info = self.stagnation_detector.get_stagnation_info()
        
        reasons = []
        
        if stagnation_info['is_stagnating']:
            reasons.append("performance_stagnation")
        
        if metrics.accuracy < 0.6:
            reasons.append("low_accuracy")
        
        if metrics.processing_time > 5.0:
            reasons.append("slow_processing")
        
        if metrics.coherence < 0.6:
            reasons.append("low_coherence")
        
        if metrics.resource_efficiency < 0.6:
            reasons.append("resource_inefficiency")
        
        return ", ".join(reasons) if reasons else "optimization_exploration"
    
    def _calculate_composite_performance(self, metrics: QueryPerformanceMetrics) -> float:
        """Calculate composite performance score"""
        return (
            0.3 * metrics.accuracy +
            0.2 * (1.0 - min(1.0, metrics.processing_time / 10.0)) +
            0.2 * metrics.coherence +
            0.15 * metrics.resource_efficiency +
            0.15 * metrics.user_satisfaction
        )
    
    def _has_outliers_in_recent_performance(self) -> bool:
        """Check if recent performance has outliers"""
        
        if len(self.performance_history) < 5:
            return False
        
        recent_scores = [self._calculate_composite_performance(m) for m in self.performance_history[-10:]]
        
        if len(recent_scores) < 3:
            return False
        
        # Simple outlier detection using IQR
        q75, q25 = np.percentile(recent_scores, [75, 25])
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = [score for score in recent_scores if score < lower_bound or score > upper_bound]
        
        return len(outliers) > 0
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics"""
        
        stagnation_info = self.stagnation_detector.get_stagnation_info()
        
        return {
            'current_loss_function': self.current_loss_config.loss_type.value,
            'current_parameters': self.current_loss_config.parameters,
            'total_adaptations': len(self.adaptation_history),
            'stagnation_info': stagnation_info,
            'recent_adaptations': [
                {
                    'timestamp': event.timestamp,
                    'from': event.previous_loss_type.value,
                    'to': event.new_loss_type.value,
                    'reason': event.trigger_reason,
                    'success': event.adaptation_success
                }
                for event in self.adaptation_history[-5:]
            ],
            'loss_function_performance': {
                loss_type.value: {
                    'usage_count': len(performances),
                    'avg_loss': np.mean([p['loss_value'] for p in performances]) if performances else 0.0,
                    'avg_computation_time': np.mean([p['computation_time'] for p in performances]) if performances else 0.0
                }
                for loss_type, performances in self.loss_function_performance.items()
            }
        }


class AdaptiveQueryLoss:
    """Main interface for adaptive query loss optimization"""
    
    def __init__(self):
        self.adaptive_engine = AdaptiveLossEngine()
        self.query_performance_tracker = {}
        self.total_optimizations = 0
        self.total_adaptations = 0
        
        logger.info("🎯 AdaptiveQueryLoss initialized with dynamic loss function adaptation")
    
    def compute_query_loss(self, predicted: np.ndarray, actual: np.ndarray, 
                          epoch: int, query_id: Optional[str] = None,
                          query_context: Optional[Dict[str, Any]] = None) -> LossEvaluationResult:
        """
        Compute adaptive loss for query optimization
        
        Args:
            predicted: Predicted query results/coordinates
            actual: Actual/target query results/coordinates
            epoch: Current optimization epoch
            query_id: Optional query identifier for tracking
            query_context: Additional context about the query
            
        Returns:
            LossEvaluationResult with loss value and gradient
        """
        
        if query_context is None:
            query_context = {}
        
        # Compute loss using adaptive engine
        result = self.adaptive_engine.compute_loss(predicted, actual, epoch, query_context)
        
        self.total_optimizations += 1
        
        # Track query-specific performance if query_id provided
        if query_id:
            if query_id not in self.query_performance_tracker:
                self.query_performance_tracker[query_id] = []
            
            self.query_performance_tracker[query_id].append({
                'epoch': epoch,
                'loss_value': result.loss_value,
                'loss_type': result.loss_type.value,
                'timestamp': time.time()
            })
        
        return result
    
    def update_query_performance(self, accuracy: float, processing_time: float,
                               coherence: float, resource_efficiency: float = 0.8,
                               user_satisfaction: float = 0.8) -> Optional[AdaptationEvent]:
        """
        Update query performance metrics and trigger adaptation if needed
        
        Args:
            accuracy: Query accuracy (0.0 to 1.0)
            processing_time: Processing time in seconds
            coherence: Cross-cube coherence (0.0 to 1.0)
            resource_efficiency: Resource utilization efficiency (0.0 to 1.0)
            user_satisfaction: User satisfaction score (0.0 to 1.0)
            
        Returns:
            AdaptationEvent if adaptation occurred, None otherwise
        """
        
        metrics = QueryPerformanceMetrics(
            accuracy=accuracy,
            processing_time=processing_time,
            coherence=coherence,
            resource_efficiency=resource_efficiency,
            user_satisfaction=user_satisfaction
        )
        
        adaptation_event = self.adaptive_engine.update_performance_and_adapt(metrics)
        
        if adaptation_event:
            self.total_adaptations += 1
        
        return adaptation_event
    
    def get_current_loss_function(self) -> Tuple[LossFunctionType, Dict[str, float]]:
        """Get current loss function type and parameters"""
        config = self.adaptive_engine.current_loss_config
        return config.loss_type, config.parameters
    
    def force_adaptation(self, target_loss_type: Optional[LossFunctionType] = None) -> AdaptationEvent:
        """Force adaptation to a specific loss function or auto-select"""
        
        with self.adaptive_engine.lock:
            previous_loss_type = self.adaptive_engine.current_loss_config.loss_type
            
            if target_loss_type is None:
                # Auto-select based on recent performance
                if self.adaptive_engine.performance_history:
                    recent_metrics = self.adaptive_engine.performance_history[-1]
                    target_loss_type = self.adaptive_engine._select_optimal_loss_function(recent_metrics)
                else:
                    target_loss_type = LossFunctionType.ADAPTIVE_HYBRID
            
            # Create adaptation event
            adaptation_event = AdaptationEvent(
                timestamp=time.time(),
                previous_loss_type=previous_loss_type,
                new_loss_type=target_loss_type,
                trigger_reason="manual_adaptation",
                performance_before=0.0  # Will be updated if metrics available
            )
            
            # Update configuration
            default_metrics = QueryPerformanceMetrics(0.5, 1.0, 0.5, 0.5, 0.5)
            self.adaptive_engine.current_loss_config = self.adaptive_engine._create_loss_config(
                target_loss_type, default_metrics
            )
            
            self.adaptive_engine.adaptation_history.append(adaptation_event)
            self.total_adaptations += 1
            
            logger.info(f"🔄 Forced adaptation: {previous_loss_type.value} -> {target_loss_type.value}")
            
            return adaptation_event
    
    def get_adaptive_loss_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptive loss statistics"""
        
        engine_stats = self.adaptive_engine.get_adaptation_stats()
        
        return {
            'total_optimizations': self.total_optimizations,
            'total_adaptations': self.total_adaptations,
            'adaptation_rate': self.total_adaptations / max(1, self.total_optimizations),
            'tracked_queries': len(self.query_performance_tracker),
            'engine_stats': engine_stats,
            'query_performance_summary': {
                query_id: {
                    'total_epochs': len(performance_list),
                    'avg_loss': np.mean([p['loss_value'] for p in performance_list]),
                    'loss_trend': [p['loss_value'] for p in performance_list[-5:]],
                    'loss_types_used': list(set(p['loss_type'] for p in performance_list))
                }
                for query_id, performance_list in list(self.query_performance_tracker.items())[:5]  # Show first 5
            }
        }


# Factory function for easy creation
def create_adaptive_query_loss() -> AdaptiveQueryLoss:
    """Create and initialize an adaptive query loss optimizer"""
    return AdaptiveQueryLoss()