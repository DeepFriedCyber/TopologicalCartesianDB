"""
Visualization tool for demonstrating Parseval's theorem and vector operations in TCDB.
This script provides interactive visualizations to help understand the mathematical concepts.
"""
import matplotlib.pyplot as plt
import numpy as np
from src.enhanced_tcdb import EnhancedTopologicalCartesianDB
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import json

def visualize_2d_parseval():
    """
    Visualize Parseval's theorem in 2D with an interactive plot.
    Shows the relationship between vector norm and projection coefficients.
    """
    # Create the database
    db = EnhancedTopologicalCartesianDB(dimensions=2)
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial vector - the 3-4-5 triangle example
    init_x, init_y = 3.0, 4.0
    
    # Standard basis vectors
    basis_x = [1, 0]
    basis_y = [0, 1]
    
    # Plot standard basis
    ax1.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue', label='x basis')
    ax1.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.2, fc='green', ec='green', label='y basis')
    
    # Plot the initial vector
    vector_arrow = ax1.arrow(0, 0, init_x, init_y, head_width=0.2, head_length=0.2, 
                            fc='red', ec='red', label='vector')
    
    # Projections onto basis vectors
    proj_x = ax1.arrow(0, 0, init_x, 0, head_width=0.1, head_length=0.1, 
                      fc='skyblue', ec='skyblue', linestyle='--', label='x projection')
    proj_y = ax1.arrow(init_x, 0, 0, init_y, head_width=0.1, head_length=0.1, 
                      fc='lightgreen', ec='lightgreen', linestyle='--', label='y projection')
    
    # Plot the right angle marker
    right_angle = patches.Arc((init_x, 0), 0.5, 0.5, 0, 90, 180, color='black', lw=1)
    ax1.add_patch(right_angle)
    
    # Grid and limits
    ax1.grid(True)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Vector and its Projections')
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Second plot for energy visualization
    # Initial values
    vector_norm_sq = init_x**2 + init_y**2
    proj_x_sq = init_x**2
    proj_y_sq = init_y**2
    
    # Bar chart data
    labels = ['||v||²', '|⟨v,x⟩|² + |⟨v,y⟩|²']
    energy_values = [vector_norm_sq, proj_x_sq + proj_y_sq]
    bar_colors = ['red', 'purple']
    
    bars = ax2.bar(labels, energy_values, color=bar_colors)
    ax2.set_ylabel('Energy (squared magnitude)')
    ax2.set_title('Parseval\'s Theorem Verification')
    
    # Add text showing the values
    equality_text = ax2.text(0.5, vector_norm_sq + 2, 
                           f'||v||² = {vector_norm_sq:.2f} = {proj_x_sq:.2f} + {proj_y_sq:.2f}', 
                           ha='center')
    
    # Add sliders for x and y
    ax_x = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_y = plt.axes([0.25, 0.05, 0.65, 0.03])
    
    slider_x = Slider(ax_x, 'x value', -5.0, 5.0, valinit=init_x)
    slider_y = Slider(ax_y, 'y value', -5.0, 5.0, valinit=init_y)
    
    def update(val):
        # Get new values from sliders
        x = slider_x.val
        y = slider_y.val
        
        # Update the vector arrow
        vector_arrow.set_data(x=0, y=0, dx=x, dy=y)
        
        # Update projections
        proj_x.set_data(x=0, y=0, dx=x, dy=0)
        proj_y.set_data(x=x, y=0, dx=0, dy=y)
        
        # Update right angle marker position
        right_angle.set_center((x, 0))
        
        # Update the energy values
        vector_norm_sq = x**2 + y**2
        proj_x_sq = x**2
        proj_y_sq = y**2
        
        # Update the bar chart
        bars[0].set_height(vector_norm_sq)
        bars[1].set_height(proj_x_sq + proj_y_sq)
        
        # Update text
        equality_text.set_text(f'||v||² = {vector_norm_sq:.2f} = {proj_x_sq:.2f} + {proj_y_sq:.2f}')
        equality_text.set_position((0.5, max(vector_norm_sq, proj_x_sq + proj_y_sq) + 2))
        
        # Check Parseval's theorem
        is_valid = db.verify_parseval_equality([x, y])
        if is_valid:
            equality_text.set_color('green')
        else:
            equality_text.set_color('red')
            
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    
    plt.show()

def visualize_query_provenance():
    """
    Visualize the query process with provenance information.
    Shows how points are examined and energy is calculated.
    """
    # Create the database
    db = EnhancedTopologicalCartesianDB(dimensions=2)
    
    # Insert some vectors
    db.insert_vector("triangle", [3.0, 4.0])
    db.insert_vector("unit_x", [1.0, 0.0])
    db.insert_vector("unit_y", [0.0, 1.0])
    db.insert_vector("far_point", [8.0, 8.0])
    
    # Query with provenance
    query_x, query_y = 0.0, 0.0
    radius = 5.0
    result = db.query_with_provenance(query_x, query_y, radius)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot query circle
    circle = plt.Circle((query_x, query_y), radius, fill=False, color='blue', 
                        linestyle='--', label='Query radius')
    ax.add_patch(circle)
    
    # Plot query point
    ax.scatter([query_x], [query_y], color='blue', s=100, label='Query point')
    
    # Plot all examined points
    for entry in result['provenance']['energy_breakdown']:
        point = entry['point']
        energy = entry['total_energy']
        in_radius = energy <= radius**2
        
        if in_radius:
            ax.scatter([point[0]], [point[1]], color='green', s=80, label='_')
        else:
            ax.scatter([point[0]], [point[1]], color='red', s=50, label='_')
            
        # Draw distance line
        ax.plot([query_x, point[0]], [query_y, point[1]], 'k--', alpha=0.3)
        
        # Annotate with distance
        mid_x = (query_x + point[0]) / 2
        mid_y = (query_y + point[1]) / 2
        ax.annotate(f"{np.sqrt(energy):.2f}", (mid_x, mid_y), fontsize=8)
    
    # Legend with custom entries
    ax.scatter([], [], color='green', s=80, label='Within radius')
    ax.scatter([], [], color='red', s=50, label='Outside radius')
    
    # Grid and limits
    ax.grid(True)
    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Query Provenance Visualization')
    ax.legend()
    ax.set_aspect('equal')
    
    # Add provenance summary
    provenance = result['provenance']
    summary = (
        f"Query point: {provenance['query_point']}\n"
        f"Radius: {provenance['radius']}\n"
        f"Points examined: {provenance['points_examined']}\n"
        f"Points found: {provenance['points_found']}\n"
        f"Parseval verified: {provenance['parseval_compliance']['verified']}"
    )
    
    plt.figtext(0.02, 0.02, summary, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()

def visualize_3d_projection():
    """
    Visualize vector projections and Parseval's theorem in 3D.
    """
    # Create the database
    db = EnhancedTopologicalCartesianDB(dimensions=3)
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Original vector
    vector = [3.0, 4.0, 5.0]
    
    # Standard basis
    basis_vectors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    
    # Origin
    origin = [0, 0, 0]
    
    # Plot basis vectors
    ax.quiver(*origin, *basis_vectors[0], color='r', label='x basis')
    ax.quiver(*origin, *basis_vectors[1], color='g', label='y basis')
    ax.quiver(*origin, *basis_vectors[2], color='b', label='z basis')
    
    # Plot original vector
    ax.quiver(*origin, *vector, color='purple', label='vector')
    
    # Plot projections
    proj_x = [vector[0], 0, 0]
    proj_y = [0, vector[1], 0]
    proj_z = [0, 0, vector[2]]
    
    ax.quiver(*origin, *proj_x, color='r', alpha=0.5, linestyle='dashed')
    ax.quiver(*origin, *proj_y, color='g', alpha=0.5, linestyle='dashed')
    ax.quiver(*origin, *proj_z, color='b', alpha=0.5, linestyle='dashed')
    
    # Calculate values for Parseval's theorem
    vector_norm_sq = sum(v**2 for v in vector)
    projections_sum_sq = sum(p**2 for p in [vector[0], vector[1], vector[2]])
    
    # Verify Parseval's theorem
    is_valid = db.verify_parseval_equality(vector)
    result = "Valid" if is_valid else "Invalid"
    
    # Add text for Parseval's theorem verification
    text = (f"Vector: {vector}\n"
            f"||v||² = {vector_norm_sq:.2f}\n"
            f"Sum of squared projections = {projections_sum_sq:.2f}\n"
            f"Parseval's theorem: {result}")
    
    ax.text(1, 1, 6, text, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Vector Projections and Parseval\'s Theorem')
    
    # Set limits
    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 5])
    ax.set_zlim([-1, 6])
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Topological Cartesian DB Visualization Tools")
    print("===========================================")
    print("\n1. 2D Parseval's Theorem Interactive Visualization")
    print("2. Query Provenance Visualization")
    print("3. 3D Vector Projection Visualization")
    
    choice = input("\nSelect a visualization (1-3): ")
    
    if choice == '1':
        visualize_2d_parseval()
    elif choice == '2':
        visualize_query_provenance()
    elif choice == '3':
        visualize_3d_projection()
    else:
        print("Invalid choice. Please run the script again and select 1-3.")
