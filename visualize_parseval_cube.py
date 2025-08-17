"""
Visualization tool for demonstrating Parseval's theorem using the ParsevalCube.
This script provides interactive visualizations to help understand the mathematical concepts.
"""
import matplotlib.pyplot as plt
import numpy as np
from core.spatial_db import TopologicalCartesianDB
from cubes.parseval_cube import ParsevalCube
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import json

def visualize_2d_parseval():
    """
    Visualize Parseval's theorem in 2D with an interactive plot.
    Shows the relationship between vector norm and projection coefficients.
    """
    # Create the database with ParsevalCube
    db = TopologicalCartesianDB(cell_size=1.0)
    parseval_cube = ParsevalCube(db, dimensions=2)
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial vector - the 3-4-5 triangle example
    init_x, init_y = 3.0, 4.0
    vector = [init_x, init_y]
    
    # Standard basis vectors
    basis = parseval_cube._create_standard_basis()
    
    # Calculate initial projections
    vector_array = np.array(vector, dtype=float)
    projections = parseval_cube._project_vector(vector_array, basis)
    proj_x, proj_y = projections
    
    # Plot the vector in first subplot
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.grid(True)
    ax1.set_title("Vector Representation")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    vector_arrow = ax1.arrow(0, 0, init_x, init_y, 
                           width=0.2, head_width=0.7, head_length=0.7, 
                           fc='blue', ec='blue', label="Vector")
    
    # Plot the standard basis
    ax1.arrow(0, 0, 1, 0, width=0.1, head_width=0.5, head_length=0.5, 
             fc='red', ec='red', label="e₁")
    ax1.arrow(0, 0, 0, 1, width=0.1, head_width=0.5, head_length=0.5, 
             fc='green', ec='green', label="e₂")
    
    # Plot projections
    proj_x_arrow = ax1.arrow(0, 0, proj_x, 0, width=0.1, head_width=0.5, head_length=0.5, 
                           fc='darkred', ec='darkred', linestyle='--')
    proj_y_arrow = ax1.arrow(0, 0, 0, proj_y, width=0.1, head_width=0.5, head_length=0.5, 
                           fc='darkgreen', ec='darkgreen', linestyle='--')
    
    # Show dashed lines to projections
    proj_line_x = ax1.plot([init_x, init_x], [init_y, 0], 'k--', alpha=0.5)[0]
    proj_line_y = ax1.plot([init_x, 0], [init_y, init_y], 'k--', alpha=0.5)[0]
    
    ax1.legend()
    
    # Energy visualization in the second subplot
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 50)
    ax2.grid(True)
    ax2.set_title("Parseval's Theorem: Energy Conservation")
    ax2.set_xlabel("Sum of squared projections")
    ax2.set_ylabel("Vector norm squared")
    
    # Initial energy values
    vector_norm_sq = np.sum(vector_array**2)
    projection_sum_sq = np.sum(projections**2)
    
    # Show the equality
    energy_text = ax2.text(5, 45, 
                         f"|v|² = {vector_norm_sq:.2f}\n∑|proj|² = {projection_sum_sq:.2f}",
                         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot the energy relationship (y=x line)
    ax2.plot([0, 50], [0, 50], 'r--', label="y=x (Perfect equality)")
    
    # Plot the current energy point
    energy_point = ax2.scatter(projection_sum_sq, vector_norm_sq, color='blue', s=100, zorder=5)
    
    # Add sliders for interactivity
    ax_x = plt.axes([0.2, 0.12, 0.65, 0.03])
    ax_y = plt.axes([0.2, 0.07, 0.65, 0.03])
    
    s_x = Slider(ax_x, 'x component', -10.0, 10.0, valinit=init_x)
    s_y = Slider(ax_y, 'y component', -10.0, 10.0, valinit=init_y)
    
    def update(val):
        x = s_x.val
        y = s_y.val
        
        # Update vector and calculations
        new_vector = [x, y]
        new_vector_array = np.array(new_vector, dtype=float)
        new_projections = parseval_cube._project_vector(new_vector_array, basis)
        new_proj_x, new_proj_y = new_projections
        
        # Update first subplot
        vector_arrow.set_data(x=0, y=0, dx=x, dy=y)
        proj_x_arrow.set_data(x=0, y=0, dx=new_proj_x, dy=0)
        proj_y_arrow.set_data(x=0, y=0, dx=0, dy=new_proj_y)
        proj_line_x.set_data([x, x], [y, 0])
        proj_line_y.set_data([x, 0], [y, y])
        
        # Update energy values
        new_vector_norm_sq = np.sum(new_vector_array**2)
        new_projection_sum_sq = np.sum(new_projections**2)
        
        # Update energy visualization
        energy_text.set_text(f"|v|² = {new_vector_norm_sq:.2f}\n∑|proj|² = {new_projection_sum_sq:.2f}")
        energy_point.set_offsets([new_projection_sum_sq, new_vector_norm_sq])
        
        # Check Parseval equality
        parseval_status = parseval_cube.verify_parseval_equality(new_vector)
        parseval_msg = "✓ Parseval verified" if parseval_status else "✗ Parseval doesn't hold"
        ax2.set_title(f"Parseval's Theorem: Energy Conservation - {parseval_msg}")
        
        fig.canvas.draw_idle()
    
    s_x.on_changed(update)
    s_y.on_changed(update)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.show()

if __name__ == "__main__":
    visualize_2d_parseval()
