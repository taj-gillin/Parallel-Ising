import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os

def load_lattice_snapshot(filename):
    """Load a lattice snapshot from the C++ output format."""
    # First, determine the lattice size from the first data row
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip().startswith('Slice'):
                row_data = line.strip().split()
                L = len(row_data)
                break
    
    # Initialize lattice with correct dimensions
    lattice = np.zeros((L, L, L), dtype=np.int32)
    current_slice = 0
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('Slice'):
                current_slice = int(line.split()[1].rstrip(':'))
                i += 1
                # Read the next L lines containing the slice data
                for j in range(L):
                    if i + j < len(lines):
                        row_data = lines[i + j].strip().split()
                        lattice[current_slice, j, :] = [int(x) for x in row_data]
                i += L  # Skip the slice data we just read
            else:
                i += 1
    return lattice

def load_simulation_data(sim_type):
    """Load all simulation data from the output directory."""
    base_dir = os.path.join('out', sim_type, '3d')
    lattice_dir = os.path.join(base_dir, 'lattice')
    
    # Load lattice snapshots
    snapshot_files = glob.glob(os.path.join(lattice_dir, 'step_*.dat'))
    if not snapshot_files:
        raise FileNotFoundError(f"No lattice snapshots found in {lattice_dir}")
    
    # Sort files numerically by the step number
    snapshot_files.sort(key=lambda f: int(f.split('step_')[-1].split('.')[0]))
    time_steps = np.array([int(f.split('step_')[-1].split('.')[0]) for f in snapshot_files])
    
    print(time_steps)
    lattice_snapshots = [load_lattice_snapshot(f) for f in snapshot_files]
    
    return time_steps, lattice_snapshots

def create_wireframe_cube_edges(L):
    """
    Create a wireframe cube of dimension L x L x L.
    We'll define the edges as line segments. Each segment will be defined by two points.
    """
    corners = [
        (0, 0, 0),
        (L-1, 0, 0),
        (L-1, L-1, 0),
        (0, L-1, 0),
        (0, 0, L-1),
        (L-1, 0, L-1),
        (L-1, L-1, L-1),
        (0, L-1, L-1)
    ]
    
    edges = [
        (corners[0], corners[1]), # bottom face
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
        
        (corners[4], corners[5]), # top face
        (corners[5], corners[6]),
        (corners[6], corners[7]),
        (corners[7], corners[4]),
        
        (corners[0], corners[4]), # vertical edges
        (corners[1], corners[5]),
        (corners[2], corners[6]),
        (corners[3], corners[7])
    ]
    
    x_coords = []
    y_coords = []
    z_coords = []
    for (x1, y1, z1), (x2, y2, z2) in edges:
        x_coords.extend([x1, x2, None])
        y_coords.extend([y1, y2, None])
        z_coords.extend([z1, z2, None])

    trace = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(color='black', width=3),
        name='Cube Edges'
    )
    return trace

def is_edge_site(x, y, z, L):
    """
    Determine if a site (x,y,z) is on an edge of the cube.
    A site is on an edge if exactly two of its coordinates are at their extremes (0 or L-1)
    and the third coordinate is not.
    """
    coords = [x, y, z]
    extremes = [c == 0 or c == L-1 for c in coords]
    # We want exactly two coordinates to be at their extremes
    if sum(extremes) != 2:
        return False
    
    # Make sure the non-extreme coordinate is not at 0 or L-1
    for i, is_extreme in enumerate(extremes):
        if not is_extreme:  # This is our non-edge coordinate
            return 0 < coords[i] < L-1
    
    return False

def create_interactive_visualization(sim_type):
    """
    Create an interactive 3D visualization showing only spins on the cube's edges along with the cube wireframe.
    """
    time_steps, lattice_snapshots = load_simulation_data(sim_type)
    
    # Determine lattice size L
    L = lattice_snapshots[0].shape[0]
    cube_trace = create_wireframe_cube_edges(L)
    
    # Create figure
    fig = go.Figure()
    
    frames = []
    for t, lattice in enumerate(lattice_snapshots):
        # Find spins on edges first
        x_pos, y_pos, z_pos = np.where(lattice == 1)
        x_neg, y_neg, z_neg = np.where(lattice == -1)
        
        # Now we can add debug prints
        if t == 0:  # Only print for first frame
            print(f"\nAnalyzing frame {t}")
            print(f"Total +1 spins: {len(x_pos)}")
            print(f"Total -1 spins: {len(x_neg)}")
        
        # Filter so we only keep edge spins
        edge_x_pos = []
        edge_y_pos = []
        edge_z_pos = []
        for i in range(len(x_pos)):
            if is_edge_site(x_pos[i], y_pos[i], z_pos[i], L):
                edge_x_pos.append(x_pos[i])
                edge_y_pos.append(y_pos[i])
                edge_z_pos.append(z_pos[i])
        
        edge_x_neg = []
        edge_y_neg = []
        edge_z_neg = []
        for i in range(len(x_neg)):
            if is_edge_site(x_neg[i], y_neg[i], z_neg[i], L):
                edge_x_neg.append(x_neg[i])
                edge_y_neg.append(y_neg[i])
                edge_z_neg.append(z_neg[i])
        
        if t == 0:  # Only print for first frame
            print(f"Edge +1 spins: {len(edge_x_pos)}")
            print(f"Edge -1 spins: {len(edge_x_neg)}")
            print("\nSample edge positions (+1 spins):")
            for i in range(min(5, len(edge_x_pos))):
                print(f"  ({edge_x_pos[i]}, {edge_y_pos[i]}, {edge_z_pos[i]})")
        
        # Create traces for edge spins
        trace_pos = go.Scatter3d(
            x=edge_x_pos,
            y=edge_y_pos,
            z=edge_z_pos,
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle'
            ),
            name='Spin +1'
        )
        
        trace_neg = go.Scatter3d(
            x=edge_x_neg,
            y=edge_y_neg,
            z=edge_z_neg,
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                symbol='circle'
            ),
            name='Spin -1'
        )
        
        # Each frame includes wireframe + spins on edges
        frame_data = [cube_trace, trace_pos, trace_neg]
        frames.append(go.Frame(data=frame_data, name=f'frame_{t}'))
    
    # Add initial data
    fig.add_trace(cube_trace)  # cube edges
    fig.add_trace(frames[0].data[1])  # spin +1 edges first frame
    fig.add_trace(frames[0].data[2])  # spin -1 edges first frame
    
    # Update layout with slider and buttons
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Time Step: '},
            'steps': [
                {
                    'method': 'animate',
                    'label': str(t),
                    'args': [[f'frame_{t}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for t in range(len(frames))
            ]
        }]
    )
    
    # Add frames to the figure
    fig.frames = frames
    
    # Save to HTML file
    output_file = os.path.join('out', sim_type, '3d', 'visualization_edges_spins.html')
    fig.write_html(output_file)
    print(f"Interactive visualization saved to {output_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize 3D Ising model lattice edges with spins')
    parser.add_argument('sim_type', help='Simulation type (e.g., mpi)')
    
    args = parser.parse_args()
    
    # Create and save visualization
    create_interactive_visualization(args.sim_type)
