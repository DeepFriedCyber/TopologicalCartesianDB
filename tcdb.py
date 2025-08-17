from mpmath import mp

class Simplex:
    def __init__(self, vertices):
        self.vertices = vertices
        mp.dps = 50  # Set high precision for exact arithmetic

    def volume(self):
        """Calculate volume using Cayley-Menger determinant"""
        n = len(self.vertices)
        if n != 4:
            raise ValueError("Only tetrahedrons supported")
        
        # Build Cayley-Menger matrix
        M = [[0]*(n+1) for _ in range(n+1)]
        M[0][0] = 0
        for i in range(1, n+1):
            M[i][0] = 1
            M[0][i] = 1
            for j in range(1, n+1):
                # Calculate squared distance between vertices i-1 and j-1
                dx = self.vertices[i-1][0] - self.vertices[j-1][0]
                dy = self.vertices[i-1][1] - self.vertices[j-1][1]
                dz = self.vertices[i-1][2] - self.vertices[j-1][2]
                M[i][j] = dx*dx + dy*dy + dz*dz
        
        # Compute determinant
        M_mp = mp.matrix(M)
        det = mp.det(M_mp)
        volume = mp.sqrt(abs(det)) / mp.factorial(n-1) / (2**((n-1)/2))
        return float(volume)

class TopologicalCartesianDB:
    def __init__(self):
        self.simplices = []
        self.data = {}
        self.index = {}
        
    def add_simplex(self, vertices, data=None):
        """Add a simplex to the database with optional associated data"""
        simplex = Simplex(vertices)
        simplex_id = len(self.simplices)
        self.simplices.append(simplex)
        
        if data:
            self.data[simplex_id] = data
            
        # Create simple index
        for vertex in vertices:
            vertex_tuple = tuple(vertex)
            if vertex_tuple not in self.index:
                self.index[vertex_tuple] = []
            self.index[vertex_tuple].append(simplex_id)
            
        return simplex_id
    
    def query_point(self, point):
        """Find simplices that contain this point"""
        point_tuple = tuple(point)
        if point_tuple in self.index:
            return self.index[point_tuple]
        return []
    
    def get_total_volume(self):
        """Calculate total volume of all simplices"""
        return sum(simplex.volume() for simplex in self.simplices)
