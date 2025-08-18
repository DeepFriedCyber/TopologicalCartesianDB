import numpy as np
from scipy.spatial import distance
from ripser import ripser
from persim import plot_diagrams

class SimplicialComplexBuilder:
    """Builder for simplicial complexes from spatial data points"""
    
    def __init__(self):
        self.distance_matrix = None
        
    def compute_distance_matrix(self, points):
        """Compute pairwise distances between points"""
        return distance.squareform(distance.pdist(points))
        
    def build_filtration(self, points, max_radius=float('inf')):
        """Build a filtration of simplicial complexes using Vietoris-Rips complex"""
        # Convert points to numpy array if they aren't already
        points_array = np.array([list(p) for p in points])
        self.distance_matrix = self.compute_distance_matrix(points_array)
        
        # We return the distance matrix which ripser will use to build the filtration
        return self.distance_matrix


class PersistentHomologyTCDB:
    """Integration of persistent homology with TCDB for multi-scale spatial analysis"""
    
    def __init__(self, base_db):
        """Initialize with a base TCDB instance"""
        self.base_db = base_db
        self.complex_builder = SimplicialComplexBuilder()
        self.persistence_results = None
        
    def build_simplicial_complex(self, max_radius=float('inf')):
        """Build filtration of simplicial complexes from spatial points"""
        # Extract points from the database
        points = list(self.base_db.points.values())
        return self.complex_builder.build_filtration(points, max_radius)
    
    def compute_persistent_homology(self, max_dimension=2, max_radius=float('inf')):
        """Compute persistent homology features"""
        distance_matrix = self.build_simplicial_complex(max_radius)
        
        # Compute persistent homology using ripser
        self.persistence_results = ripser(
            distance_matrix, 
            maxdim=max_dimension, 
            distance_matrix=True
        )
        
        return self.persistence_results
    
    def get_betti_numbers(self):
        """Extract Betti numbers from persistence diagram"""
        if self.persistence_results is None:
            raise ValueError("Compute persistent homology first")
        
        betti_numbers = []
        for dim, diagram in enumerate(self.persistence_results["dgms"]):
            # Count points with persistence above threshold
            persistence_threshold = 0.1  # Can be adjusted
            persistent_points = [p for p in diagram if p[1] - p[0] > persistence_threshold]
            betti_numbers.append(len(persistent_points))
            
        return betti_numbers
    
    def extract_features(self):
        """Extract topological features from persistence diagrams"""
        if self.persistence_results is None:
            raise ValueError("Compute persistent homology first")
            
        features = {}
        for dim, diagram in enumerate(self.persistence_results["dgms"]):
            if len(diagram) == 0:
                continue
                
            # Calculate persistence (difference between birth and death)
            persistence = np.array([p[1] - p[0] for p in diagram if p[1] < float('inf')])
            
            if len(persistence) > 0:
                features[f"dim_{dim}_max_persistence"] = np.max(persistence)
                features[f"dim_{dim}_mean_persistence"] = np.mean(persistence)
                features[f"dim_{dim}_sum_persistence"] = np.sum(persistence)
                features[f"dim_{dim}_count"] = len(persistence)
        
        return features
    
    def visualize_persistence_diagram(self, title="Persistence Diagram"):
        """Visualize the persistence diagram"""
        if self.persistence_results is None:
            raise ValueError("Compute persistent homology first")
            
        plot_diagrams(self.persistence_results["dgms"], title=title)

    def query_by_homology(self, target_betti_numbers, tolerance=0.2):
        """Find regions with similar homological features"""
        # This is a simplified version - a real implementation would likely
        # need to compute homology over different regions of the data
        current_betti = self.get_betti_numbers()
        
        # Check if current betti numbers match target within tolerance
        match = True
        for i, (current, target) in enumerate(zip(current_betti, target_betti_numbers)):
            if abs(current - target) > tolerance * target:
                match = False
                break
                
        return match