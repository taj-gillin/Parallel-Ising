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
    # Construct the full path
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

def create_interactive_visualization(sim_type):
    """
    Create an interactive 3D visualization of lattice evolution over time using Plotly.
    
    Parameters
    ----------
    sim_type : str
        Simulation type (e.g., 'mpi')
    output_file : str
        Path to save the HTML output file
    """
    time_steps, lattice_snapshots = load_simulation_data(sim_type)
    
    # Create figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # Create frames for animation
    frames = []
    for t, lattice in enumerate(lattice_snapshots):
        frame_data = []
        
        # Add points for spin +1 (red)
        x1, y1, z1 = np.where(lattice == 1)
        trace1 = go.Scatter3d(
            x=x1, y=y1, z=z1,
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Spin +1',
            visible=True
        )
        
        # Add points for spin -1 (blue)
        x2, y2, z2 = np.where(lattice == -1)
        trace2 = go.Scatter3d(
            x=x2, y=y2, z=z2,
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Spin -1',
            visible=True
        )
        
        frame_data.extend([trace1, trace2])
        frames.append(go.Frame(data=frame_data, name=f'frame_{t}'))
    
    # Add initial data
    fig.add_trace(frames[0].data[0])
    fig.add_trace(frames[0].data[1])
    
    # Update layout with slider and buttons
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
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
    output_file = os.path.join('out', sim_type, '3d', 'visualization.html')
    fig.write_html(output_file)
    print(f"Interactive visualization saved to {output_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize 3D Ising model lattice evolution')
    parser.add_argument('sim_type', help='Simulation type (e.g., mpi)')
    
    args = parser.parse_args()

    
    # Create and save visualization
    create_interactive_visualization(args.sim_type)
