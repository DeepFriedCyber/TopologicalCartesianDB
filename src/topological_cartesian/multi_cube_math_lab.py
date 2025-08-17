#!/usr/bin/env python3
"""
Multi-Cube Mathematical Model Laboratory

Revolutionary approach: Each cube tests different mathematical models,
enabling parallel learning and cross-pollination of best techniques.

This creates a "mathematical evolution" system where cubes compete
and collaborate to find optimal models for different domains.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)


class MathModelType(Enum):
    """Types of mathematical models for cube specialization"""
    MANIFOLD_LEARNING = "manifold_learning"
    GRAPH_THEORY = "graph_theory"
    TENSOR_DECOMPOSITION = "tensor_decomposition"
    INFORMATION_THEORY = "information_theory"
    GEOMETRIC_DEEP_LEARNING = "geometric_deep_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    TOPOLOGICAL_DATA_ANALYSIS = "topological_data_analysis"
    QUANTUM_INSPIRED = "quantum_inspired"
    PERSISTENT_HOMOLOGY = "persistent_homology"  # NEW: Advanced topological analysis
    HYBRID_TOPOLOGICAL_BAYESIAN = "hybrid_topological_bayesian"  # TDD Phase 3: Hybrid model


@dataclass
class MathModelExperiment:
    """Individual mathematical model experiment"""
    model_type: MathModelType
    cube_name: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = False
    error_message: str = ""
    improvement_score: float = 0.0
    hybrid_components: Optional[Dict[str, Any]] = None  # TDD Phase 3: Track hybrid model components


@dataclass
class CubeMathSpecialization:
    """Mathematical specialization for each cube"""
    cube_name: str
    primary_model: MathModelType
    secondary_models: List[MathModelType] = field(default_factory=list)
    expertise_domain: str = ""
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.1
    best_performance: float = 0.0
    experiment_history: List[MathModelExperiment] = field(default_factory=list)


class MultiCubeMathLaboratory:
    """Laboratory for testing mathematical models across multiple cubes"""
    
    def __init__(self):
        self.cube_specializations = self._initialize_cube_specializations()
        self.global_performance_tracker = {}
        self.cross_cube_knowledge = {}
        self.evolution_generation = 0
        
        # TDD Phase 3: Advanced Integration Features
        self.enable_hybrid_models = False
        self.enable_cross_cube_learning = False
        self.cross_cube_learner = None
        self.hybrid_model_factory = None
        
        logger.info("🧪 Multi-Cube Mathematical Laboratory initialized")
    
    def _initialize_cube_specializations(self) -> Dict[str, CubeMathSpecialization]:
        """Initialize mathematical specializations for each cube"""
        
        specializations = {
            # CODE_CUBE: Focus on structural and algorithmic models
            'code_cube': CubeMathSpecialization(
                cube_name='code_cube',
                primary_model=MathModelType.GRAPH_THEORY,
                secondary_models=[MathModelType.TOPOLOGICAL_DATA_ANALYSIS, MathModelType.TENSOR_DECOMPOSITION],
                expertise_domain='structural_analysis',
                learning_rate=0.02
            ),
            
            # DATA_CUBE: Focus on statistical and information-theoretic models
            'data_cube': CubeMathSpecialization(
                cube_name='data_cube',
                primary_model=MathModelType.INFORMATION_THEORY,
                secondary_models=[MathModelType.MANIFOLD_LEARNING, MathModelType.BAYESIAN_OPTIMIZATION],
                expertise_domain='statistical_analysis',
                learning_rate=0.015
            ),
            
            # USER_CUBE: Focus on behavioral and geometric models
            'user_cube': CubeMathSpecialization(
                cube_name='user_cube',
                primary_model=MathModelType.MANIFOLD_LEARNING,
                secondary_models=[MathModelType.GEOMETRIC_DEEP_LEARNING, MathModelType.GRAPH_THEORY],
                expertise_domain='behavioral_modeling',
                learning_rate=0.025
            ),
            
            # SYSTEM_CUBE: Focus on optimization and performance models
            'system_cube': CubeMathSpecialization(
                cube_name='system_cube',
                primary_model=MathModelType.BAYESIAN_OPTIMIZATION,
                secondary_models=[MathModelType.QUANTUM_INSPIRED, MathModelType.TENSOR_DECOMPOSITION],
                expertise_domain='performance_optimization',
                learning_rate=0.01
            ),
            
            # TEMPORAL_CUBE: Focus on time-series and dynamic models
            'temporal_cube': CubeMathSpecialization(
                cube_name='temporal_cube',
                primary_model=MathModelType.PERSISTENT_HOMOLOGY,  # NEW: Advanced topological analysis
                secondary_models=[MathModelType.TOPOLOGICAL_DATA_ANALYSIS, MathModelType.MANIFOLD_LEARNING, MathModelType.INFORMATION_THEORY],
                expertise_domain='temporal_dynamics',
                learning_rate=0.02
            )
        }
        
        logger.info(f"🧊 Initialized {len(specializations)} cube mathematical specializations")
        return specializations
    
    def create_math_model_experiments(self, coordinates: Dict[str, np.ndarray]) -> List[MathModelExperiment]:
        """Create experiments for each cube's mathematical models"""
        
        experiments = []
        
        for cube_name, specialization in self.cube_specializations.items():
            if cube_name not in coordinates:
                continue
            
            coords = coordinates[cube_name]
            
            # Primary model experiment
            primary_exp = MathModelExperiment(
                model_type=specialization.primary_model,
                cube_name=cube_name,
                parameters=self._get_model_parameters(specialization.primary_model, coords, is_primary=True)
            )
            experiments.append(primary_exp)
            
            # Secondary model experiments
            for secondary_model in specialization.secondary_models:
                secondary_exp = MathModelExperiment(
                    model_type=secondary_model,
                    cube_name=cube_name,
                    parameters=self._get_model_parameters(secondary_model, coords, is_primary=False)
                )
                experiments.append(secondary_exp)
            
            # TDD Phase 3: Add hybrid model experiments when enabled
            if self.enable_hybrid_models:
                hybrid_exp = MathModelExperiment(
                    model_type=MathModelType.HYBRID_TOPOLOGICAL_BAYESIAN,
                    cube_name=cube_name,
                    parameters=self._get_model_parameters(MathModelType.HYBRID_TOPOLOGICAL_BAYESIAN, coords, is_primary=False)
                )
                experiments.append(hybrid_exp)
        
        logger.info(f"🔬 Created {len(experiments)} mathematical model experiments")
        return experiments
    
    def _get_model_parameters(self, model_type: MathModelType, coordinates: np.ndarray, is_primary: bool = True) -> Dict[str, Any]:
        """Get parameters for specific mathematical model"""
        
        n_samples, n_features = coordinates.shape
        
        base_params = {
            'n_samples': n_samples,
            'n_features': n_features,
            'is_primary': is_primary
        }
        
        if model_type == MathModelType.MANIFOLD_LEARNING:
            return {
                **base_params,
                'method': 'tsne' if n_samples < 50 else 'pca',
                'n_components': min(32, n_features),
                'perplexity': min(30, n_samples // 3) if n_samples > 10 else 5
            }
        
        elif model_type == MathModelType.GRAPH_THEORY:
            return {
                **base_params,
                'similarity_threshold': 0.6 if is_primary else 0.7,
                'community_method': 'spectral',
                'centrality_measures': ['betweenness', 'closeness', 'pagerank']
            }
        
        elif model_type == MathModelType.INFORMATION_THEORY:
            return {
                **base_params,
                'entropy_method': 'histogram',
                'bins': min(50, n_samples // 2),
                'optimization_method': 'gradient_descent'
            }
        
        elif model_type == MathModelType.TENSOR_DECOMPOSITION:
            return {
                **base_params,
                'decomposition_method': 'cp',
                'rank': min(20, min(n_samples, n_features)),
                'max_iterations': 100
            }
        
        elif model_type == MathModelType.TOPOLOGICAL_DATA_ANALYSIS:
            return {
                **base_params,
                'max_dimension': 2,
                'filtration_method': 'rips',
                'max_edge_length': 2.0
            }
        
        elif model_type == MathModelType.BAYESIAN_OPTIMIZATION:
            return {
                **base_params,
                'acquisition_function': 'expected_improvement',
                'kernel': 'rbf',
                'n_iterations': 50
            }
        
        elif model_type == MathModelType.GEOMETRIC_DEEP_LEARNING:
            return {
                **base_params,
                'architecture': 'gcn',
                'hidden_dims': [64, 32],
                'learning_rate': 0.01
            }
        
        elif model_type == MathModelType.QUANTUM_INSPIRED:
            return {
                **base_params,
                'algorithm': 'qaoa',
                'n_layers': 3,
                'optimization_steps': 100
            }
        
        elif model_type == MathModelType.HYBRID_TOPOLOGICAL_BAYESIAN:
            return {
                **base_params,
                'fusion_strategy': 'adaptive',
                'topological_weight': 0.5,
                'bayesian_weight': 0.5,
                'synergy_bonus': True,
                'domain': 'default'
            }
        
        else:
            return base_params
    
    def run_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> MathModelExperiment:
        """Run a single mathematical model experiment"""
        
        start_time = time.time()
        
        try:
            if experiment.model_type == MathModelType.MANIFOLD_LEARNING:
                result = self._run_manifold_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.GRAPH_THEORY:
                result = self._run_graph_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.INFORMATION_THEORY:
                result = self._run_information_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.TENSOR_DECOMPOSITION:
                result = self._run_tensor_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.TOPOLOGICAL_DATA_ANALYSIS:
                result = self._run_tda_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.BAYESIAN_OPTIMIZATION:
                result = self._run_bayesian_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.GEOMETRIC_DEEP_LEARNING:
                result = self._run_geometric_dl_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.QUANTUM_INSPIRED:
                result = self._run_quantum_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.PERSISTENT_HOMOLOGY:
                result = self._run_persistent_homology_experiment(experiment, coordinates)
            elif experiment.model_type == MathModelType.HYBRID_TOPOLOGICAL_BAYESIAN:
                result = self._run_hybrid_topological_bayesian_experiment(experiment, coordinates)
            else:
                result = self._run_baseline_experiment(experiment, coordinates)
            
            experiment.performance_metrics = result
            experiment.success = True
            experiment.improvement_score = self._calculate_improvement_score(result)
            
        except Exception as e:
            experiment.success = False
            experiment.error_message = str(e)
            experiment.improvement_score = 0.0
            logger.warning(f"⚠️ Experiment failed: {experiment.cube_name}/{experiment.model_type.value} - {e}")
        
        experiment.execution_time = time.time() - start_time
        return experiment
    
    def _run_manifold_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run manifold learning experiment"""
        
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        
        method = experiment.parameters['method']
        n_components = experiment.parameters['n_components']
        
        if method == 'tsne' and coordinates.shape[0] > experiment.parameters['perplexity']:
            model = TSNE(n_components=min(n_components, 3), 
                        perplexity=experiment.parameters['perplexity'],
                        random_state=42)
            embedding = model.fit_transform(coordinates)
        else:
            model = PCA(n_components=n_components)
            embedding = model.fit_transform(coordinates)
        
        # Calculate metrics
        if embedding.shape[0] > 2:
            silhouette = silhouette_score(coordinates, np.arange(coordinates.shape[0]) % 3)
            embedding_silhouette = silhouette_score(embedding, np.arange(embedding.shape[0]) % 3)
        else:
            silhouette = 0.5
            embedding_silhouette = 0.5
        
        return {
            'dimensionality_reduction': (coordinates.shape[1] - embedding.shape[1]) / coordinates.shape[1],
            'silhouette_improvement': embedding_silhouette - silhouette,
            'variance_explained': 0.8 if method == 'pca' else 0.6,
            'embedding_quality': embedding_silhouette
        }
    
    def _run_graph_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run graph theory experiment"""
        
        import networkx as nx
        from scipy.spatial.distance import pdist, squareform
        from sklearn.cluster import SpectralClustering
        
        # Build similarity graph
        similarities = 1 - pdist(coordinates, metric='cosine')
        similarity_matrix = squareform(similarities)
        
        threshold = experiment.parameters['similarity_threshold']
        
        # Create graph
        graph = nx.Graph()
        n_points = coordinates.shape[0]
        
        for i in range(n_points):
            graph.add_node(i)
        
        edges_added = 0
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if similarity_matrix[i, j] > threshold:
                    graph.add_edge(i, j, weight=similarity_matrix[i, j])
                    edges_added += 1
        
        # Calculate metrics
        if len(graph.nodes) > 0 and len(graph.edges) > 0:
            density = len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1) / 2)
            
            # Community detection
            if len(graph.edges) > 0:
                clustering = SpectralClustering(n_clusters=min(5, len(graph.nodes)), random_state=42)
                try:
                    adj_matrix = nx.adjacency_matrix(graph)
                    labels = clustering.fit_predict(adj_matrix.toarray())
                    n_communities = len(set(labels))
                except:
                    n_communities = 1
            else:
                n_communities = 1
            
            # Centrality measures
            try:
                centralities = nx.betweenness_centrality(graph)
                avg_centrality = np.mean(list(centralities.values()))
            except:
                avg_centrality = 0.0
        else:
            density = 0.0
            n_communities = 1
            avg_centrality = 0.0
        
        return {
            'graph_density': density,
            'community_modularity': n_communities / max(1, len(graph.nodes)),
            'centrality_distribution': avg_centrality,
            'connectivity_score': edges_added / max(1, n_points * (n_points - 1) / 2)
        }
    
    def _run_information_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run information theory experiment"""
        
        from scipy.stats import entropy
        
        # Calculate entropy
        flat_coords = coordinates.flatten()
        hist, _ = np.histogram(flat_coords, bins=experiment.parameters['bins'], density=True)
        hist = hist + 1e-10  # Avoid log(0)
        
        coord_entropy = entropy(hist)
        
        # Mutual information between dimensions
        mutual_infos = []
        for i in range(coordinates.shape[1]):
            for j in range(i + 1, coordinates.shape[1]):
                # Simplified mutual information calculation
                hist_2d, _, _ = np.histogram2d(coordinates[:, i], coordinates[:, j], bins=20)
                hist_2d = hist_2d + 1e-10
                hist_2d = hist_2d / np.sum(hist_2d)
                
                # Marginal entropies
                hist_i = np.sum(hist_2d, axis=1)
                hist_j = np.sum(hist_2d, axis=0)
                
                # Mutual information
                mi = 0.0
                for x in range(len(hist_i)):
                    for y in range(len(hist_j)):
                        if hist_2d[x, y] > 0:
                            mi += hist_2d[x, y] * np.log(hist_2d[x, y] / (hist_i[x] * hist_j[y]))
                
                mutual_infos.append(mi)
        
        avg_mutual_info = np.mean(mutual_infos) if mutual_infos else 0.0
        
        return {
            'coordinate_entropy': coord_entropy,
            'average_mutual_information': avg_mutual_info,
            'information_density': coord_entropy / coordinates.size,
            'redundancy_score': 1.0 - (avg_mutual_info / max(coord_entropy, 1e-10))
        }
    
    def _run_tensor_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run tensor decomposition experiment"""
        
        # Create a simple 3D tensor from coordinates
        n_samples, n_features = coordinates.shape
        
        # Reshape into tensor-like structure
        if n_features >= 3:
            tensor_shape = (min(10, n_samples), min(5, n_features), min(3, n_features))
            tensor = np.random.randn(*tensor_shape) * 0.1 + coordinates[:tensor_shape[0], :tensor_shape[1], np.newaxis]
        else:
            tensor = coordinates.reshape(n_samples, n_features, 1)
        
        # Simple tensor decomposition using SVD
        unfolded = tensor.reshape(tensor.shape[0], -1)
        U, s, Vt = np.linalg.svd(unfolded, full_matrices=False)
        
        rank = min(experiment.parameters['rank'], len(s))
        reconstruction_error = np.sum(s[rank:] ** 2) / np.sum(s ** 2) if len(s) > rank else 0.0
        
        return {
            'reconstruction_error': reconstruction_error,
            'compression_ratio': rank / len(s),
            'singular_value_decay': s[0] / s[-1] if len(s) > 1 else 1.0,
            'tensor_rank_efficiency': 1.0 - reconstruction_error
        }
    
    def _run_tda_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run topological data analysis experiment"""
        
        # Simplified TDA using distance-based features
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import entropy
        
        distances = pdist(coordinates)
        distance_matrix = squareform(distances)
        
        # Estimate topological features
        max_distance = np.max(distances)
        persistence_threshold = max_distance * 0.5
        
        # Count connected components at different scales
        thresholds = np.linspace(0, max_distance, 10)
        n_components = []
        
        for threshold in thresholds:
            # Simple connected components counting
            adjacency = distance_matrix < threshold
            # This is a simplified version - real TDA would use proper persistent homology
            n_comp = np.sum(np.diag(adjacency))  # Simplified estimate
            n_components.append(n_comp)
        
        # Calculate persistence features
        persistence_entropy = entropy(np.array(n_components) + 1e-10)
        topological_complexity = np.std(n_components)
        
        return {
            'persistence_entropy': persistence_entropy,
            'topological_complexity': topological_complexity,
            'betti_0_estimate': np.mean(n_components),
            'homology_persistence': np.max(n_components) - np.min(n_components)
        }
    
    def _run_bayesian_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run Bayesian optimization experiment"""
        
        # Simplified Bayesian optimization simulation
        from scipy.optimize import minimize
        
        def objective_function(x):
            # Simple quadratic objective based on coordinates
            if len(x) != coordinates.shape[1]:
                x = x[:coordinates.shape[1]]
            
            distances = np.linalg.norm(coordinates - x, axis=1)
            return np.mean(distances)
        
        # Random starting point
        x0 = np.mean(coordinates, axis=0)
        
        # Optimize
        result = minimize(objective_function, x0, method='L-BFGS-B')
        
        return {
            'optimization_success': float(result.success),
            'function_evaluations': result.nfev,
            'final_objective': result.fun,
            'convergence_rate': 1.0 / max(result.nfev, 1)
        }
    
    def _run_geometric_dl_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run geometric deep learning experiment (simplified)"""
        
        # Simplified geometric features
        n_samples, n_features = coordinates.shape
        
        # Calculate geometric properties
        centroid = np.mean(coordinates, axis=0)
        distances_to_centroid = np.linalg.norm(coordinates - centroid, axis=1)
        
        # Estimate curvature (simplified)
        if n_samples > 2:
            # Use second derivatives as curvature estimate
            sorted_indices = np.argsort(distances_to_centroid)
            sorted_coords = coordinates[sorted_indices]
            
            if len(sorted_coords) > 2:
                first_diff = np.diff(sorted_coords, axis=0)
                second_diff = np.diff(first_diff, axis=0)
                curvature_estimate = np.mean(np.linalg.norm(second_diff, axis=1))
            else:
                curvature_estimate = 0.0
        else:
            curvature_estimate = 0.0
        
        return {
            'geometric_spread': np.std(distances_to_centroid),
            'curvature_estimate': curvature_estimate,
            'geometric_complexity': np.mean(np.std(coordinates, axis=0)),
            'manifold_dimension_estimate': min(n_features, 3)  # Simplified estimate
        }
    
    def _run_quantum_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run quantum-inspired experiment"""
        
        # Quantum-inspired features using superposition and entanglement concepts
        n_samples, n_features = coordinates.shape
        
        # Simulate quantum superposition using linear combinations
        superposition_states = []
        for i in range(min(5, n_samples)):
            # Create superposition of coordinate states
            weights = np.random.rand(n_samples)
            weights = weights / np.sum(weights)  # Normalize
            superposition = np.sum(weights[:, np.newaxis] * coordinates, axis=0)
            superposition_states.append(superposition)
        
        superposition_states = np.array(superposition_states)
        
        # Estimate "entanglement" using correlations
        if len(superposition_states) > 1:
            correlation_matrix = np.corrcoef(superposition_states)
            entanglement_measure = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        else:
            entanglement_measure = 0.0
        
        # Quantum coherence estimate
        coherence = np.mean(np.abs(np.fft.fft(coordinates.flatten())))
        
        return {
            'quantum_superposition_diversity': np.std(superposition_states.flatten()),
            'entanglement_measure': entanglement_measure,
            'quantum_coherence': coherence,
            'quantum_advantage_estimate': min(entanglement_measure * coherence, 1.0)
        }
    
    def _run_persistent_homology_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run persistent homology experiment"""
        
        try:
            # Try enhanced persistent homology first
            from .enhanced_persistent_homology import create_enhanced_persistent_homology_experiment
            
            # Determine domain based on cube name
            domain_mapping = {
                'temporal_cube': 'pandemic',
                'data_cube': 'scientific',
                'code_cube': 'default',
                'user_cube': 'default',
                'system_cube': 'default'
            }
            domain = domain_mapping.get(experiment.cube_name, 'default')
            
            # Run the enhanced persistent homology experiment
            ph_result = create_enhanced_persistent_homology_experiment(
                coordinates, 
                experiment.cube_name, 
                experiment.parameters,
                texts=None,  # No texts available in this context
                domain=domain
            )
            
            if ph_result['success']:
                return ph_result['performance_metrics']
            else:
                # Fallback to original persistent homology
                return self._run_original_persistent_homology(coordinates, experiment)
                
        except ImportError:
            # Fallback to original persistent homology
            return self._run_original_persistent_homology(coordinates, experiment)
    
    def _run_original_persistent_homology(self, coordinates: np.ndarray, experiment: MathModelExperiment) -> Dict[str, float]:
        """Run original persistent homology as fallback"""
        try:
            # Import the original persistent homology model
            from .persistent_homology_model import create_persistent_homology_experiment
            
            # Run the original persistent homology experiment
            ph_result = create_persistent_homology_experiment(
                coordinates, 
                experiment.cube_name, 
                experiment.parameters
            )
            
            if ph_result['success']:
                return ph_result['performance_metrics']
            else:
                # Final fallback to simplified persistent homology
                return self._run_simplified_persistent_homology(coordinates)
                
        except ImportError:
            # Final fallback if persistent homology model not available
            return self._run_simplified_persistent_homology(coordinates)
    
    def _run_simplified_persistent_homology(self, coordinates: np.ndarray) -> Dict[str, float]:
        """Simplified persistent homology when full implementation not available"""
        
        from scipy.spatial.distance import pdist, squareform
        
        # Compute distance matrix
        distances = pdist(coordinates)
        distance_matrix = squareform(distances)
        
        # Multi-scale analysis
        max_distance = np.max(distances)
        thresholds = np.linspace(0, max_distance, 10)
        
        # Track connected components at different scales
        component_counts = []
        for threshold in thresholds:
            # Create adjacency matrix
            adjacency = distance_matrix <= threshold
            
            # Count connected components using simple DFS
            n_points = coordinates.shape[0]
            visited = np.zeros(n_points, dtype=bool)
            components = 0
            
            def dfs(node):
                visited[node] = True
                for neighbor in range(n_points):
                    if adjacency[node, neighbor] and not visited[neighbor]:
                        dfs(neighbor)
            
            for i in range(n_points):
                if not visited[i]:
                    dfs(i)
                    components += 1
            
            component_counts.append(components)
        
        # Compute persistence-like features
        component_changes = np.diff(component_counts)
        persistence_score = np.sum(np.abs(component_changes))
        stability_score = 1.0 - (np.std(component_counts) / (np.mean(component_counts) + 1e-8))
        
        # Betti numbers (simplified)
        final_components = component_counts[-1]
        betti_0 = final_components
        
        return {
            'betti_0': betti_0,
            'betti_1': 0,  # Simplified: assume no 1D holes
            'betti_2': 0,  # Simplified: assume no 2D holes
            'total_persistence': persistence_score,
            'max_persistence': max(component_changes) if len(component_changes) > 0 else 0.0,
            'stability_score': stability_score,
            'feature_vector_norm': np.sqrt(betti_0**2 + persistence_score**2 + stability_score**2)
        }
    
    def _run_hybrid_topological_bayesian_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """TDD Phase 3: Run hybrid topological-Bayesian experiment"""
        
        try:
            # Import the hybrid model
            from .enhanced_persistent_homology import HybridTopologicalBayesianModel
            
            # Create hybrid model
            hybrid_model = HybridTopologicalBayesianModel()
            
            # Run hybrid analysis
            domain = experiment.parameters.get('domain', 'default')
            result = hybrid_model.compute_hybrid_analysis(coordinates, domain=domain)
            
            if result['success']:
                # Store hybrid components for tracking
                experiment.hybrid_components = {
                    'topological_score': result['topological_score'],
                    'bayesian_score': result['bayesian_score'],
                    'fusion_weights': result['fusion_weights'],
                    'topological_result': result.get('topological_result')
                }
                
                # Convert to performance metrics format
                return {
                    'hybrid_score': result['hybrid_score'],
                    'topological_score': result['topological_score'],
                    'bayesian_score': result['bayesian_score'],
                    'topological_weight': result['fusion_weights']['topological_weight'],
                    'bayesian_weight': result['fusion_weights']['bayesian_weight'],
                    'synergy_bonus': result['hybrid_score'] - (
                        result['topological_score'] * result['fusion_weights']['topological_weight'] +
                        result['bayesian_score'] * result['fusion_weights']['bayesian_weight']
                    )
                }
            else:
                # Fallback to simplified hybrid scoring
                return self._run_simplified_hybrid_experiment(coordinates)
                
        except ImportError:
            # Fallback if hybrid model not available
            return self._run_simplified_hybrid_experiment(coordinates)
    
    def _run_simplified_hybrid_experiment(self, coordinates: np.ndarray) -> Dict[str, float]:
        """Simplified hybrid experiment as fallback"""
        
        # Simple combination of basic topological and optimization metrics
        n_samples, n_features = coordinates.shape
        
        # Simple topological score (based on data structure)
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        avg_distance = np.mean(distances[distances > 0])
        topological_score = 1.0 / (1.0 + avg_distance)  # Inverse distance as structure measure
        
        # Simple Bayesian score (based on dimensionality and size)
        bayesian_score = min(n_features / 100.0, 1.0) * (1.0 - abs(n_samples - 50) / 100.0)
        bayesian_score = max(0.1, bayesian_score)
        
        # Adaptive fusion
        if n_features >= 50:
            topological_weight, bayesian_weight = 0.3, 0.7
        else:
            topological_weight, bayesian_weight = 0.7, 0.3
        
        # Hybrid score with synergy
        weighted_score = topological_score * topological_weight + bayesian_score * bayesian_weight
        synergy_bonus = min(topological_score, bayesian_score) * 0.2
        hybrid_score = weighted_score + synergy_bonus
        
        return {
            'hybrid_score': hybrid_score,
            'topological_score': topological_score,
            'bayesian_score': bayesian_score,
            'topological_weight': topological_weight,
            'bayesian_weight': bayesian_weight,
            'synergy_bonus': synergy_bonus
        }
    
    def _run_baseline_experiment(self, experiment: MathModelExperiment, coordinates: np.ndarray) -> Dict[str, float]:
        """Run baseline experiment"""
        
        return {
            'baseline_variance': np.var(coordinates),
            'baseline_mean': np.mean(coordinates),
            'baseline_std': np.std(coordinates),
            'baseline_range': np.max(coordinates) - np.min(coordinates)
        }
    
    def _calculate_improvement_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall improvement score from metrics"""
        
        # Weight different metrics based on importance
        weights = {
            'dimensionality_reduction': 0.2,
            'silhouette_improvement': 0.3,
            'graph_density': 0.15,
            'community_modularity': 0.15,
            'coordinate_entropy': 0.1,
            'reconstruction_error': -0.2,  # Negative because lower is better
            'optimization_success': 0.25,
            'geometric_complexity': 0.1,
            'quantum_advantage_estimate': 0.15,
            # Persistent homology metrics
            'stability_score': 0.3,
            'total_persistence': 0.2,
            'max_persistence': 0.15,
            'betti_0': 0.1,
            'betti_1': 0.1,
            'betti_2': 0.1,
            'feature_vector_norm': 0.1,
            # TDD Phase 3: Hybrid model metrics
            'hybrid_score': 0.4,
            'topological_score': 0.2,
            'bayesian_score': 0.2,
            'synergy_bonus': 0.2
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]
                score += weight * value
                total_weight += abs(weight)
        
        return score / max(total_weight, 1.0)
    
    def run_parallel_experiments(self, coordinates: Dict[str, np.ndarray], max_workers: int = 5) -> Dict[str, List[MathModelExperiment]]:
        """Run all experiments in parallel across cubes"""
        
        logger.info("🚀 Starting parallel mathematical model experiments...")
        
        # TDD Phase 3: Initialize cross-cube learner if enabled
        if self.enable_cross_cube_learning and self.cross_cube_learner is None:
            from .enhanced_persistent_homology import CrossCubeTopologicalLearner
            self.cross_cube_learner = CrossCubeTopologicalLearner()
        
        experiments = self.create_math_model_experiments(coordinates)
        results = {cube_name: [] for cube_name in self.cube_specializations.keys()}
        
        # Group experiments by cube for parallel execution
        cube_experiments = {}
        for exp in experiments:
            if exp.cube_name not in cube_experiments:
                cube_experiments[exp.cube_name] = []
            cube_experiments[exp.cube_name].append(exp)
        
        # Run experiments in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_experiment = {}
            
            for cube_name, cube_exps in cube_experiments.items():
                if cube_name in coordinates:
                    coords = coordinates[cube_name]
                    for exp in cube_exps:
                        future = executor.submit(self.run_experiment, exp, coords)
                        future_to_experiment[future] = (cube_name, exp)
            
            # Collect results
            for future in as_completed(future_to_experiment):
                cube_name, original_exp = future_to_experiment[future]
                try:
                    completed_exp = future.result()
                    results[cube_name].append(completed_exp)
                    
                    # Update cube specialization
                    self.cube_specializations[cube_name].experiment_history.append(completed_exp)
                    
                    if completed_exp.success and completed_exp.improvement_score > self.cube_specializations[cube_name].best_performance:
                        self.cube_specializations[cube_name].best_performance = completed_exp.improvement_score
                    
                    logger.info(f"✅ {cube_name}/{completed_exp.model_type.value}: score={completed_exp.improvement_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Experiment failed: {cube_name}/{original_exp.model_type.value} - {e}")
        
        logger.info(f"🎉 Completed {sum(len(exps) for exps in results.values())} experiments")
        
        # TDD Phase 3: Apply cross-cube learning if enabled
        if self.enable_cross_cube_learning and self.cross_cube_learner is not None:
            self._apply_cross_cube_learning(results, coordinates)
        
        return results
    
    def _apply_cross_cube_learning(self, results: Dict[str, List[MathModelExperiment]], coordinates: Dict[str, np.ndarray]):
        """TDD Phase 3: Apply cross-cube learning to improve performance"""
        
        try:
            from .enhanced_persistent_homology import TopologicalPatternExtractor
            
            # Extract patterns from successful experiments
            pattern_extractor = TopologicalPatternExtractor()
            all_experiments = []
            
            for cube_name, experiments in results.items():
                for exp in experiments:
                    if exp.success and exp.improvement_score > 0:
                        # Convert experiment to pattern format
                        experiment_data = {
                            'domain': 'default',  # Could be enhanced to detect domain
                            'improvement_score': exp.improvement_score,
                            'parameters': exp.parameters,
                            'topological_features': {
                                'betti_numbers': exp.performance_metrics.get('betti_numbers', {0: 1, 1: 0, 2: 0}),
                                'stability_score': exp.performance_metrics.get('stability_score', 0.5)
                            }
                        }
                        all_experiments.append(experiment_data)
            
            # Extract successful patterns
            if all_experiments:
                patterns = pattern_extractor.extract_successful_patterns(all_experiments)
                self.cross_cube_learner.learned_patterns = patterns
                
                logger.info(f"🧠 Cross-cube learner extracted {len(patterns)} successful patterns")
            
        except Exception as e:
            logger.warning(f"⚠️ Cross-cube learning failed: {e}")
    
    def has_learned_patterns(self) -> bool:
        """Check if cross-cube learner has learned patterns"""
        if hasattr(self, 'cross_cube_learner') and self.cross_cube_learner is not None:
            return hasattr(self.cross_cube_learner, 'learned_patterns') and len(self.cross_cube_learner.learned_patterns) > 0
        return False
    
    def analyze_cross_cube_learning(self, experiment_results: Dict[str, List[MathModelExperiment]]) -> Dict[str, Any]:
        """Analyze learning patterns across cubes"""
        
        logger.info("🧠 Analyzing cross-cube learning patterns...")
        
        analysis = {
            'best_models_per_cube': {},
            'model_performance_ranking': {},
            'cross_cube_knowledge_transfer': {},
            'evolution_recommendations': {}
        }
        
        # Find best models for each cube
        for cube_name, experiments in experiment_results.items():
            successful_experiments = [exp for exp in experiments if exp.success]
            if successful_experiments:
                best_exp = max(successful_experiments, key=lambda x: x.improvement_score)
                analysis['best_models_per_cube'][cube_name] = {
                    'model_type': best_exp.model_type.value,
                    'score': best_exp.improvement_score,
                    'metrics': best_exp.performance_metrics
                }
        
        # Rank model types globally
        model_scores = {}
        for experiments in experiment_results.values():
            for exp in experiments:
                if exp.success:
                    model_type = exp.model_type.value
                    if model_type not in model_scores:
                        model_scores[model_type] = []
                    model_scores[model_type].append(exp.improvement_score)
        
        for model_type, scores in model_scores.items():
            analysis['model_performance_ranking'][model_type] = {
                'average_score': np.mean(scores),
                'std_score': np.std(scores),
                'success_rate': len(scores) / sum(len(exps) for exps in experiment_results.values()),
                'best_score': np.max(scores)
            }
        
        # Knowledge transfer opportunities
        for cube_name, specialization in self.cube_specializations.items():
            if cube_name in analysis['best_models_per_cube']:
                best_model = analysis['best_models_per_cube'][cube_name]['model_type']
                best_score = analysis['best_models_per_cube'][cube_name]['score']
                
                # Find other cubes that could benefit from this model
                transfer_opportunities = []
                for other_cube, other_best in analysis['best_models_per_cube'].items():
                    if other_cube != cube_name and other_best['score'] < best_score:
                        transfer_opportunities.append({
                            'target_cube': other_cube,
                            'potential_improvement': best_score - other_best['score'],
                            'recommended_model': best_model
                        })
                
                analysis['cross_cube_knowledge_transfer'][cube_name] = transfer_opportunities
        
        # Evolution recommendations
        for cube_name, specialization in self.cube_specializations.items():
            recommendations = []
            
            if cube_name in experiment_results:
                cube_experiments = experiment_results[cube_name]
                successful_experiments = [exp for exp in cube_experiments if exp.success]
                
                if successful_experiments:
                    # Recommend promoting best secondary model to primary
                    secondary_experiments = [exp for exp in successful_experiments 
                                           if exp.model_type != specialization.primary_model]
                    
                    if secondary_experiments:
                        best_secondary = max(secondary_experiments, key=lambda x: x.improvement_score)
                        if best_secondary.improvement_score > specialization.best_performance * 1.1:
                            recommendations.append({
                                'action': 'promote_secondary_to_primary',
                                'model': best_secondary.model_type.value,
                                'expected_improvement': best_secondary.improvement_score - specialization.best_performance
                            })
                    
                    # Recommend new secondary models from global best performers
                    global_best_models = sorted(analysis['model_performance_ranking'].items(), 
                                              key=lambda x: x[1]['average_score'], reverse=True)[:3]
                    
                    for model_name, model_stats in global_best_models:
                        model_type = MathModelType(model_name)
                        if (model_type != specialization.primary_model and 
                            model_type not in specialization.secondary_models):
                            recommendations.append({
                                'action': 'add_secondary_model',
                                'model': model_name,
                                'expected_improvement': model_stats['average_score']
                            })
            
            analysis['evolution_recommendations'][cube_name] = recommendations
        
        return analysis
    
    def evolve_cube_specializations(self, analysis: Dict[str, Any]) -> Dict[str, CubeMathSpecialization]:
        """Evolve cube specializations based on learning analysis"""
        
        logger.info("🧬 Evolving cube mathematical specializations...")
        
        evolved_specializations = {}
        
        for cube_name, specialization in self.cube_specializations.items():
            evolved_spec = CubeMathSpecialization(
                cube_name=cube_name,
                primary_model=specialization.primary_model,
                secondary_models=specialization.secondary_models.copy(),
                expertise_domain=specialization.expertise_domain,
                learning_rate=specialization.learning_rate,
                adaptation_threshold=specialization.adaptation_threshold,
                best_performance=specialization.best_performance,
                experiment_history=specialization.experiment_history.copy()
            )
            
            # Apply evolution recommendations
            if cube_name in analysis['evolution_recommendations']:
                recommendations = analysis['evolution_recommendations'][cube_name]
                
                for rec in recommendations:
                    if rec['action'] == 'promote_secondary_to_primary':
                        new_primary = MathModelType(rec['model'])
                        # Move current primary to secondary
                        if evolved_spec.primary_model not in evolved_spec.secondary_models:
                            evolved_spec.secondary_models.append(evolved_spec.primary_model)
                        evolved_spec.primary_model = new_primary
                        # Remove new primary from secondary models
                        if new_primary in evolved_spec.secondary_models:
                            evolved_spec.secondary_models.remove(new_primary)
                        
                        logger.info(f"🔄 {cube_name}: Promoted {rec['model']} to primary model")
                    
                    elif rec['action'] == 'add_secondary_model':
                        new_secondary = MathModelType(rec['model'])
                        if new_secondary not in evolved_spec.secondary_models:
                            evolved_spec.secondary_models.append(new_secondary)
                            # Keep only top 3 secondary models
                            if len(evolved_spec.secondary_models) > 3:
                                evolved_spec.secondary_models = evolved_spec.secondary_models[:3]
                        
                        logger.info(f"➕ {cube_name}: Added {rec['model']} as secondary model")
            
            evolved_specializations[cube_name] = evolved_spec
        
        self.evolution_generation += 1
        logger.info(f"🧬 Evolution generation {self.evolution_generation} completed")
        
        return evolved_specializations
    
    def generate_learning_report(self, experiment_results: Dict[str, List[MathModelExperiment]], 
                               analysis: Dict[str, Any]) -> str:
        """Generate comprehensive learning report"""
        
        report = f"""
🧪 MULTI-CUBE MATHEMATICAL MODEL LABORATORY REPORT
{'='*60}
Generation: {self.evolution_generation}
Total Experiments: {sum(len(exps) for exps in experiment_results.values())}

📊 CUBE PERFORMANCE SUMMARY:
{'-'*40}
"""
        
        for cube_name, best_model_info in analysis['best_models_per_cube'].items():
            report += f"""
🧊 {cube_name.upper()}:
   Best Model: {best_model_info['model_type']}
   Score: {best_model_info['score']:.3f}
   Specialization: {self.cube_specializations[cube_name].expertise_domain}
"""
        
        report += f"""
🏆 GLOBAL MODEL RANKING:
{'-'*40}
"""
        
        sorted_models = sorted(analysis['model_performance_ranking'].items(), 
                             key=lambda x: x[1]['average_score'], reverse=True)
        
        for i, (model_name, stats) in enumerate(sorted_models[:5], 1):
            report += f"""
{i}. {model_name}:
   Average Score: {stats['average_score']:.3f} ± {stats['std_score']:.3f}
   Success Rate: {stats['success_rate']:.1%}
   Best Score: {stats['best_score']:.3f}
"""
        
        report += f"""
🔄 KNOWLEDGE TRANSFER OPPORTUNITIES:
{'-'*40}
"""
        
        for cube_name, transfers in analysis['cross_cube_knowledge_transfer'].items():
            if transfers:
                report += f"""
{cube_name} can share knowledge with:
"""
                for transfer in transfers[:3]:  # Top 3 opportunities
                    report += f"   • {transfer['target_cube']}: +{transfer['potential_improvement']:.3f} improvement\n"
        
        report += f"""
🧬 EVOLUTION RECOMMENDATIONS:
{'-'*40}
"""
        
        for cube_name, recommendations in analysis['evolution_recommendations'].items():
            if recommendations:
                report += f"""
{cube_name}:
"""
                for rec in recommendations[:2]:  # Top 2 recommendations
                    report += f"   • {rec['action']}: {rec['model']} (+{rec['expected_improvement']:.3f})\n"
        
        report += f"""
🎯 NEXT STEPS:
{'-'*40}
1. Apply evolution recommendations
2. Run next generation experiments
3. Implement cross-cube knowledge transfer
4. Monitor performance improvements

🚀 The mathematical evolution continues!
"""
        
        return report


def create_multi_cube_math_lab() -> MultiCubeMathLaboratory:
    """Factory function to create multi-cube mathematical laboratory"""
    return MultiCubeMathLaboratory()


if __name__ == "__main__":
    # Example usage
    lab = create_multi_cube_math_lab()
    
    # Create test data
    test_coordinates = {
        'code_cube': np.random.randn(50, 5),
        'data_cube': np.random.randn(45, 5),
        'user_cube': np.random.randn(40, 5),
        'system_cube': np.random.randn(35, 5),
        'temporal_cube': np.random.randn(30, 5)
    }
    
    # Run experiments
    results = lab.run_parallel_experiments(test_coordinates)
    
    # Analyze and evolve
    analysis = lab.analyze_cross_cube_learning(results)
    evolved_specs = lab.evolve_cube_specializations(analysis)
    
    # Generate report
    report = lab.generate_learning_report(results, analysis)
    print(report)
    
    print("🧪 Multi-Cube Mathematical Laboratory demonstration completed!")