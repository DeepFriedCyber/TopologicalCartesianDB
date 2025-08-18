import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import distance, KDTree
import networkx as nx
import os
import time
from datetime import datetime

# Ensure numpy is imported
if 'np' not in globals():
    import numpy as np

class HodgeLaplacianCalculator:
    """Calculator for Hodge Laplacians on simplicial complexes"""
    
    def __init__(self):
        self.boundary_matrices = {}
        self.laplacians = {}
    
    def compute_boundary_matrix(self, simplices_k, simplices_k_minus_1, dim_k):
        """Compute boundary matrix from k-simplices to (k-1)-simplices"""
        rows = []
        cols = []
        data = []
        
        # Map each (k-1)-simplex to its index
        simplex_to_idx = {tuple(sorted(s)): i for i, s in enumerate(simplices_k_minus_1)}
        
        # For each k-simplex, compute its boundary
        for i, simplex in enumerate(simplices_k):
            for j in range(dim_k + 1):
                # Remove j-th vertex to get a face (a (k-1)-simplex)
                face = tuple(sorted([simplex[k] for k in range(dim_k + 1) if k != j]))
                if face in simplex_to_idx:
                    rows.append(simplex_to_idx[face])
                    cols.append(i)
                    # Determine the sign based on the orientation
                    sign = (-1)**j
                    data.append(sign)
                    
        return sparse.csr_matrix((data, (rows, cols)), 
                               shape=(len(simplices_k_minus_1), len(simplices_k)))
    
    def compute(self, simplicial_complex):
        """Compute all Hodge Laplacians for the simplicial complex"""
        max_dim = max(simplicial_complex.keys())
        
        # Compute boundary matrices
        for k in range(1, max_dim + 1):
            self.boundary_matrices[k] = self.compute_boundary_matrix(
                simplicial_complex[k],
                simplicial_complex[k-1],
                k
            )
        
        # Compute Hodge Laplacians
        for k in range(max_dim + 1):
            # L_k = B_{k+1} * B_{k+1}^T + B_k^T * B_k
            L_k = None
            
            # First term: B_{k+1} * B_{k+1}^T (if k+1 simplices exist)
            if k < max_dim:
                B_kp1 = self.boundary_matrices[k+1]
                L_k = B_kp1 @ B_kp1.transpose()
            
            # Second term: B_k^T * B_k (if k simplices exist)
            if k > 0:
                B_k = self.boundary_matrices[k]
                second_term = B_k.transpose() @ B_k
                if L_k is None:
                    L_k = second_term
                else:
                    L_k = L_k + second_term
            
            # Handle the case k=0 and no higher simplices
            if L_k is None:
                n_vertices = len(simplicial_complex[0])
                L_k = sparse.csr_matrix((n_vertices, n_vertices))
            
            self.laplacians[k] = L_k
            
        return self.laplacians


def build_simplicial_complex(points, k_neighbors=5, max_dim=2):
    """Build a simplicial complex from point cloud data using k-nearest neighbors"""
    n_points = len(points)
    complex = {0: [[i] for i in range(n_points)]}  # 0-simplices (vertices)
    
    # Build k-NN graph
    kdtree = KDTree(points)
    edges = []
    for i in range(n_points):
        # Find k nearest neighbors
        distances, indices = kdtree.query(points[i], k=min(k_neighbors+1, n_points))
        # First index is the point itself, skip it
        for j in indices[1:]:
            # Ensure we don't add edges twice
            if i < j:
                edges.append((i, j))
    
    # 1-simplices (edges)
    complex[1] = [sorted(edge) for edge in edges]
    
    # Higher-dimensional simplices
    if max_dim >= 2:
        # Create a graph from edges to find triangles (2-simplices)
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Find 2-simplices (triangles)
        triangles = []
        for i, j in edges:
            # Find common neighbors of i and j
            i_neighbors = set(G.neighbors(i))
            j_neighbors = set(G.neighbors(j))
            common_neighbors = i_neighbors.intersection(j_neighbors)
            
            for k in common_neighbors:
                # Ensure we don't add triangles multiple times
                triangle = sorted([i, j, k])
                if triangle not in triangles:
                    triangles.append(triangle)
        
        complex[2] = triangles
    
    return complex


def compute_eigenvalues(laplacians, k=10):
    """Compute first k eigenvalues of each Laplacian matrix"""
    eigenvalues = {}
    
    for dim, laplacian in laplacians.items():
        size = laplacian.shape[0]
        if size == 0:  # Skip empty laplacians
            eigenvalues[dim] = np.array([])
            continue
            
        # Use sparse eigenvalue solver to compute k smallest eigenvalues
        # If matrix size is smaller than k, compute all eigenvalues
        k_eff = min(k, size - 1)
        if k_eff <= 0:
            eigenvalues[dim] = np.array([])
            continue
            
        try:
            vals, _ = eigsh(laplacian, k=k_eff, which='SM')
            eigenvalues[dim] = np.sort(vals)
        except:
            # Fall back to dense computation for very small matrices
            dense_lap = laplacian.toarray()
            vals = np.linalg.eigvalsh(dense_lap)
            eigenvalues[dim] = np.sort(vals)[:k_eff]
    
    return eigenvalues


def detect_spectral_anomalies(eigenvalues, threshold=2.0):
    """Detect anomalies based on spectral properties"""
    anomalies = []
    
    # Focus on the first non-trivial eigenvalue of the 0-dimensional Laplacian
    # This corresponds to the algebraic connectivity (Fiedler value)
    if 0 in eigenvalues and len(eigenvalues[0]) > 1:
        fiedler_values = eigenvalues[0][1:]  # Skip the first eigenvalue (should be zero)
        
        # Use spectral gap as an anomaly indicator
        if len(fiedler_values) > 0:
            spectral_gaps = np.diff(fiedler_values)
            mean_gap = np.mean(spectral_gaps)
            std_gap = np.std(spectral_gaps)
            
            # Calculate an anomaly score for each point
            # Initially set all points to score 0
            point_scores = np.zeros(len(spectral_gaps) + 1)
            
            # Identify anomalies as points where spectral gap exceeds threshold
            for i, gap in enumerate(spectral_gaps):
                # Normalize the gap as a z-score
                z_score = (gap - mean_gap) / std_gap if std_gap > 0 else 0
                
                # Store the z-score as the anomaly score for this point
                point_scores[i] = max(0, z_score)
                
                # Flag points with high scores as anomalies
                if z_score > threshold:
                    anomalies.append(i)
    else:
        # Return empty point_scores array if no eigenvalues are available
        point_scores = np.array([])
    
    # Return as a tuple to make it clear we're returning multiple values
    return (anomalies, point_scores)


def compute_anomaly_scores(eigenvalues):
    """Compute anomaly scores based on spectral properties"""
    scores = {}
    
    for dim, vals in eigenvalues.items():
        if len(vals) <= 1:
            continue
            
        # Skip the zero eigenvalue for dim=0
        start_idx = 1 if dim == 0 else 0
        
        if len(vals) > start_idx:
            non_zero_vals = vals[start_idx:]
            
            # Compute spectral gap score
            if len(non_zero_vals) > 1:
                gaps = np.diff(non_zero_vals)
                mean_gap = np.mean(gaps)
                
                # Normalize gaps
                if mean_gap > 0:
                    norm_gaps = gaps / mean_gap
                    scores[f'dim_{dim}_spectral_gaps'] = norm_gaps.tolist()
            
            # Compute eigenvalue magnitude score
            scores[f'dim_{dim}_eigenvalues'] = non_zero_vals.tolist()
    
    return scores


class HodgeAnomalyTCDB:
    """Integration of Hodge Laplacian-based anomaly detection with TCDB"""
    
    def __init__(self, base_db=None):
        """Initialize with an optional base TCDB instance"""
        self.base_db = base_db
        self.hodge_calculator = HodgeLaplacianCalculator()
        self.last_results = None
    
    def detect_spatial_anomalies(self, k_neighbors=5, max_dim=2, eigenvalue_threshold=2.0, points=None):
        """Detect anomalies in spatial distribution using Hodge Laplacians"""
        # Extract points from the database or use provided points
        if points is None:
            if self.base_db is None:
                raise ValueError("Either base_db must be initialized or points must be provided")
            point_ids = list(self.base_db.points.keys())
            points = np.array(list(self.base_db.points.values()))
        else:
            point_ids = list(range(len(points)))
            points = np.array(points)

        # Build simplicial complex from spatial points
        complex = build_simplicial_complex(points, k_neighbors, max_dim)
        
        # Compute Hodge Laplacians
        hodge_laplacians = self.hodge_calculator.compute(complex)
        
        # Extract spectral features
        eigenvalues = compute_eigenvalues(hodge_laplacians)
        
        # Detect anomalies based on spectral properties
        # Explicitly unpack the tuple returned by detect_spectral_anomalies
        anomaly_indices, point_scores = detect_spectral_anomalies(eigenvalues, eigenvalue_threshold)
        anomalous_points = [point_ids[idx] for idx in anomaly_indices]
        
        # Compute anomaly scores
        anomaly_scores = compute_anomaly_scores(eigenvalues)
        
        # Add point scores to anomaly scores dictionary
        anomaly_scores['point_scores'] = point_scores.tolist() if isinstance(point_scores, np.ndarray) else point_scores

        results = {
            'anomalous_points': anomalous_points,
            'anomaly_indices': anomaly_indices,
            'anomaly_scores': anomaly_scores,
            'spectral_features': {dim: vals.tolist() for dim, vals in eigenvalues.items()}
        }
        
        self.last_results = results
        return results
    
    def get_harmonic_components(self, dim=1):
        """Extract harmonic components (null space of Hodge Laplacian)
           These represent the non-trivial homology generators"""
        # First ensure we have computed the Laplacians
        if not hasattr(self.hodge_calculator, 'laplacians') or not self.hodge_calculator.laplacians:
            raise ValueError("Must run detect_spatial_anomalies first")
        
        if dim not in self.hodge_calculator.laplacians:
            return None
        
        laplacian = self.hodge_calculator.laplacians[dim]
        
        # Compute the null space (harmonic components)
        # We look for eigenvalues close to zero
        try:
            vals, vecs = eigsh(laplacian, k=min(6, laplacian.shape[0]-1), which='SM')
            harmonic_idx = np.where(np.abs(vals) < 1e-10)[0]
            harmonic_components = vecs[:, harmonic_idx]
            
            return {
                'dimension': dim,
                'betti_number': harmonic_components.shape[1],
                'harmonic_components': harmonic_components
            }
        except:
            return {
                'dimension': dim,
                'betti_number': 0,
                'harmonic_components': np.array([])
            }
    
    def localize_anomalies(self, anomaly_indices=None, radius=None, points=None):
        """Localize the detected anomalies to specific regions"""
        if anomaly_indices is None and self.last_results:
            anomaly_indices = self.last_results['anomaly_indices']
            
        if not anomaly_indices:
            return []
            
        if points is None:
            if self.base_db is None:
                raise ValueError("Either base_db must be initialized or points must be provided")
            points = np.array(list(self.base_db.points.values()))
            
        # Determine radius if not specified (use average nearest neighbor distance)
        if radius is None:
            kdtree = KDTree(points)
            distances, _ = kdtree.query(points, k=min(2, len(points)))  # Get distance to nearest neighbor
            radius = np.mean(distances[:, 1]) * 2 if distances.shape[1] > 1 else 0.1  # Double the average NN distance
        
        # For each anomaly, find points within its radius
        regions = []
        for idx in anomaly_indices:
            if 0 <= idx < len(points):
                center = points[idx]
                kdtree = KDTree(points)
                indices = kdtree.query_ball_point(center, radius)
                regions.append(indices)
            
        return regions


# Cybersecurity Extension: Network and Filesystem Anomaly Detection
class CyberHodgeDetector:
    """Application of Hodge Laplacian-based anomaly detection for cybersecurity"""
    
    def __init__(self):
        self.detector = HodgeAnomalyTCDB()
        self.baseline = None
        self.last_scan_time = None
    
    def _network_to_points(self, network_data):
        """Convert network connection data to a point cloud
        
        network_data: List of dictionaries containing connection information
        Each dict should have source_ip, dest_ip, port, timestamp, bytes_transferred
        
        Returns: numpy array of points where each point encodes a connection
        """
        points = []
        for conn in network_data:
            # Convert IP to a numeric value (simplified)
            src_ip_parts = list(map(int, conn['source_ip'].split('.')))
            dst_ip_parts = list(map(int, conn['dest_ip'].split('.')))
            
            # Create a feature vector: [src_ip_numeric, dst_ip_numeric, port, time, bytes]
            src_ip_num = src_ip_parts[0]*256**3 + src_ip_parts[1]*256**2 + src_ip_parts[2]*256 + src_ip_parts[3]
            dst_ip_num = dst_ip_parts[0]*256**3 + dst_ip_parts[1]*256**2 + dst_ip_parts[2]*256 + dst_ip_parts[3]
            
            # Normalize the values
            normalized_src = src_ip_num / (256**4)
            normalized_dst = dst_ip_num / (256**4)
            normalized_port = conn['port'] / 65535
            
            # Convert timestamp to seconds since epoch and normalize by day
            if isinstance(conn['timestamp'], str):
                timestamp = time.mktime(datetime.strptime(conn['timestamp'], "%Y-%m-%d %H:%M:%S").timetuple())
            else:
                timestamp = conn['timestamp']
            normalized_time = (timestamp % (24*3600)) / (24*3600)  # Time within day
            
            # Normalize bytes transferred (using log scale)
            normalized_bytes = np.log1p(conn['bytes_transferred']) / 30  # Assuming max log(bytes) around 30
            
            points.append([normalized_src, normalized_dst, normalized_port, normalized_time, normalized_bytes])
        
        return np.array(points)
    
    def _filesystem_to_points(self, root_dir):
        """Convert filesystem structure to a point cloud
        
        root_dir: Root directory to scan
        
        Returns: 
        - points: numpy array where each point represents a file
        - file_paths: list of file paths corresponding to each point
        """
        points = []
        file_paths = []
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                
                try:
                    # Get file stats
                    stats = os.stat(full_path)
                    
                    # Features: [path_depth, file_size, modified_time, access_time, dir_position]
                    path_depth = len(rel_path.split(os.sep))
                    file_size = stats.st_size
                    modified_time = stats.st_mtime
                    access_time = stats.st_atime
                    
                    # Position in directory (based on alphabetical ordering)
                    dir_files = sorted(filenames)
                    position = dir_files.index(filename) / max(1, len(dir_files))
                    
                    # Normalize values
                    norm_depth = path_depth / 20  # Assuming max depth of 20
                    norm_size = np.log1p(file_size) / 30  # Log scale, assuming max log(size) around 30
                    
                    # Time since epoch in days, normalized by 30 days
                    now = time.time()
                    norm_mod_time = min(1.0, (now - modified_time) / (30 * 24 * 3600))
                    norm_acc_time = min(1.0, (now - access_time) / (30 * 24 * 3600))
                    
                    points.append([norm_depth, norm_size, norm_mod_time, norm_acc_time, position])
                    file_paths.append(full_path)
                except (FileNotFoundError, PermissionError):
                    # Skip files we can't access
                    continue
        
        return np.array(points), file_paths
    
    def establish_network_baseline(self, network_data, k_neighbors=5):
        """Establish a baseline for normal network behavior"""
        points = self._network_to_points(network_data)
        results = self.detector.detect_spatial_anomalies(
            k_neighbors=k_neighbors,
            max_dim=2,
            eigenvalue_threshold=2.0,
            points=points
        )
        
        self.baseline = {
            'spectral_features': results['spectral_features'],
            'time': datetime.now()
        }
        self.last_scan_time = datetime.now()
        
        return self.baseline
    
    def detect_network_anomalies(self, network_data, k_neighbors=5, threshold=2.5):
        """Detect anomalies in network connections"""
        points = self._network_to_points(network_data)
        results = self.detector.detect_spatial_anomalies(
            k_neighbors=k_neighbors,
            max_dim=2,
            eigenvalue_threshold=threshold,
            points=points
        )
        
        # Map anomalies back to network connections
        anomalous_connections = []
        for idx in results['anomaly_indices']:
            if 0 <= idx < len(network_data):
                anomalous_connections.append(network_data[idx])
        
        self.last_scan_time = datetime.now()
        
        return {
            'anomalous_connections': anomalous_connections,
            'anomaly_indices': results['anomaly_indices'],
            'anomaly_scores': results['anomaly_scores'],
            'total_connections': len(network_data)
        }
    
    def scan_filesystem(self, root_dir, k_neighbors=5, threshold=2.5):
        """Scan a filesystem for anomalous files based on metadata patterns"""
        points, file_paths = self._filesystem_to_points(root_dir)
        
        # Skip if not enough files
        if len(points) < k_neighbors + 1:
            return {
                'anomalous_files': [],
                'anomaly_indices': [],
                'total_files': len(points)
            }
        
        results = self.detector.detect_spatial_anomalies(
            k_neighbors=k_neighbors,
            max_dim=2,
            eigenvalue_threshold=threshold,
            points=points
        )
        
        # Map anomalies back to files
        anomalous_files = []
        for idx in results['anomaly_indices']:
            if 0 <= idx < len(file_paths):
                anomalous_files.append(file_paths[idx])
        
        self.last_scan_time = datetime.now()
        
        return {
            'anomalous_files': anomalous_files,
            'anomaly_indices': results['anomaly_indices'],
            'anomaly_scores': results['anomaly_scores'],
            'total_files': len(file_paths)
        }
    
    def explain_anomaly(self, anomaly_idx, network_data=None, file_paths=None):
        """
        Explain why an entity was flagged as anomalous using multiple topological perspectives.
        
        Args:
            anomaly_idx: Index of the anomalous entity in the original dataset
            network_data: Original network connection data (if applicable)
            file_paths: Original file paths (if applicable)
            
        Returns:
            str: Multi-layered explanation of why the entity was flagged as anomalous,
                 including spectral and harmonic analysis when available
        """
        # Initialize variables at the beginning to avoid "possibly unbound" errors
        anomaly_score = 0.0
        has_point_score = False
        has_spectral_explanation = False
        has_harmonic_explanation = False
        
        # Basic validation
        if self.detector.last_results is None:
            return "No anomaly detection has been performed yet."
            
        anomaly_indices = self.detector.last_results.get('anomaly_indices', [])
        if anomaly_idx not in anomaly_indices:
            return f"Index {anomaly_idx} was not flagged as anomalous."
        
        # Prepare explanation components
        explanation_parts = []
        
        # APPROACH 1: Point-based anomaly score
        try:
            if 'anomaly_scores' in self.detector.last_results:
                point_scores = self.detector.last_results['anomaly_scores'].get('point_scores', [])
                if point_scores and anomaly_idx < len(point_scores):
                    anomaly_score = float(point_scores[anomaly_idx])
                    has_point_score = True
        except (IndexError, TypeError, ValueError, KeyError):
            pass  # Continue to other methods if this fails
        
        # APPROACH 2: Spectral gap explanation
        try:
            if 'anomaly_scores' in self.detector.last_results:
                spectral_gaps = self.detector.last_results['anomaly_scores'].get('dim_0_spectral_gaps', [])
                if spectral_gaps:
                    # Find the largest spectral gap
                    spectral_gaps_array = np.array(spectral_gaps)
                    max_gap_idx = int(np.argmax(spectral_gaps_array))
                    max_gap = float(spectral_gaps_array[max_gap_idx])
                    
                    # Only add this explanation if it's significant
                    if max_gap > 1.5:  # Threshold for considering a gap significant
                        spectral_explanation = (
                            f"The data exhibits a significant spectral gap of {max_gap:.2f} at position {max_gap_idx}, "
                            f"indicating a natural separation in the dataset's structure."
                        )
                        explanation_parts.append(spectral_explanation)
                        has_spectral_explanation = True
        except (IndexError, TypeError, ValueError, KeyError):
            pass  # Continue to other methods if this fails
        
        # APPROACH 3: Harmonic component analysis
        try:
            # Get harmonic components to explain the anomaly
            harmonic_data = self.detector.get_harmonic_components(dim=1)
            if harmonic_data and harmonic_data['betti_number'] > 0:
                components = harmonic_data['harmonic_components']
                if anomaly_idx < components.shape[0]:
                    # Get the point's contribution to the harmonic space
                    point_contribution = np.abs(components[anomaly_idx])
                    harmonic_score = float(np.max(point_contribution))
                    
                    # Only add this explanation if the contribution is significant
                    if harmonic_score > 0.1:  # Threshold for considering a contribution significant
                        harmonic_explanation = (
                            f"This entity has a significant contribution ({harmonic_score:.2f}) to the "
                            f"harmonic space, indicating it forms part of a topological cycle "
                            f"(such as a hole or loop in the data)."
                        )
                        explanation_parts.append(harmonic_explanation)
                        has_harmonic_explanation = True
                        
                        # If we don't have a point score yet, use the harmonic score
                        if not has_point_score:
                            anomaly_score = harmonic_score
                            has_point_score = True
        except Exception:
            pass  # Harmonic analysis is optional, continue if it fails
        
        # Generate the primary explanation based on the entity type
        primary_explanation = ""
        try:
            if network_data and anomaly_idx < len(network_data):
                conn = network_data[anomaly_idx]
                primary_explanation = (
                    f"Connection {conn['source_ip']} → {conn['dest_ip']}:{conn['port']} "
                    f"{f'with anomaly score {anomaly_score:.2f} ' if has_point_score else ''}"
                    f"was flagged as anomalous.\n"
                )
            elif file_paths and anomaly_idx < len(file_paths):
                file_path = file_paths[anomaly_idx]
                primary_explanation = (
                    f"File {file_path} "
                    f"{f'with anomaly score {anomaly_score:.2f} ' if has_point_score else ''}"
                    f"was flagged as anomalous.\n"
                )
            else:
                primary_explanation = (
                    f"Entity at index {anomaly_idx} "
                    f"{f'with anomaly score {anomaly_score:.2f} ' if has_point_score else ''}"
                    f"was flagged as anomalous.\n"
                )
        except (KeyError, IndexError, TypeError) as e:
            primary_explanation = (
                f"Entity at index {anomaly_idx} was flagged as anomalous.\n"
            )
        
        # Add the primary explanation to the top
        explanation_parts.insert(0, primary_explanation)
        
        # Add a generic topological explanation if we don't have specific ones
        if not (has_spectral_explanation or has_harmonic_explanation):
            generic_explanation = (
                "This entity forms a topological pattern that differs significantly "
                "from the majority of the data."
            )
            explanation_parts.append(generic_explanation)
        
        # Combine all explanation parts
        return "\n".join(explanation_parts)