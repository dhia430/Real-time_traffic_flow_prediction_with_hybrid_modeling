import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import logging
from src.traffic.ctm_model import CellTransmissionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_ctm(config_path: str, num_cells: int = 20, time_steps: int = 100):
    """
    Simulates a traffic density shockwave using the CTM class.
    Initially, we have zero density, then we inject a high density burst.
    """
    if not os.path.exists(config_path):
        logging.error(f"Config path invalid: {config_path}")
        return
        
    ctm = CellTransmissionModel(config_path, num_cells=num_cells)
    
    # Store evolution for heatmap
    evolution = np.zeros((time_steps, num_cells))
    
    logging.info("Simulating CTM shockwave...")
    for t in range(time_steps):
        # Inject density burst in the first 20 steps
        input_rho = 0.12 if (20 <= t <= 40) else 0.02
        densities = ctm.update(input_rho)
        evolution[t, :] = densities
        
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(evolution.T, aspect='auto', cmap='YlOrRd', origin='lower')
    plt.colorbar(label='Density (veh/m)')
    plt.title("CTM Density Propagation (Shockwave Simulation)")
    plt.xlabel("Time Step")
    plt.ylabel("Cell Index (Space)")
    
    os.makedirs('outputs/verification', exist_ok=True)
    save_path = 'outputs/verification/ctm_shockwave.png'
    plt.savefig(save_path)
    plt.close()
    logging.info(f"CTM shockwave simulation saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--cells', type=int, default=20, help='Number of cells')
    parser.add_argument('--steps', type=int, default=100, help='Number of timesteps')
    args = parser.parse_args()
    
    verify_ctm(args.config, args.cells, args.steps)
