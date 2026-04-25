import yaml
import numpy as np
import os
import matplotlib.pyplot as plt

class CellTransmissionModel:
    """
    Implements the macroscopic Cell Transmission Model (CTM) for predicting
    traffic density shockwaves across sequential road segments.
    """
    def __init__(self, config_path: str = 'config/config.yaml', num_cells: int = 10):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        ctm_conf = config.get('ctm', {})
        self.dx = ctm_conf.get('cell_length_meters', 20.0)
        self.dt = ctm_conf.get('time_step_seconds', 1.0)
        self.v_f = ctm_conf.get('free_flow_speed_m_s', 15.0)
        self.rho_max = ctm_conf.get('jam_density_veh_m', 0.15)
        self.q_max = ctm_conf.get('max_flow_veh_s', 0.5)
        
        # Verify Courant-Friedrichs-Lewy (CFL) Condition
        cfl = self.v_f * self.dt / self.dx
        if cfl > 1.0:
            import logging
            logging.warning(
                f"CFL condition violated: v_f*dt/dx ({cfl}) > 1.0! "
                "Numerical instability expected. Reduce dt or increase cell length."
            )
            
        self.num_cells = num_cells
        # Initial density state across all cells
        self.densities = np.zeros(self.num_cells)
        
    def _flow(self, density: float) -> float:
        """
        Calculates flow based on triangular Fundamental Diagram of traffic.
        Flow q = min(v_f * rho, w * (rho_max - rho)) up to maximum capacity q_max
        """
        free_flow = self.v_f * density
        congested_flow = self.q_max * (1.0 - (density / self.rho_max))
        # It's bounded by 0 and max flow
        q = min(free_flow, congested_flow)
        return max(0.0, q)
        
    def update(self, observed_rho_entering: float) -> np.ndarray:
        """
        Updates the CTM state based on input density at first cell.
        
        Args:
            observed_rho_entering: observed density (veh/m) entering the grid.
            
        Returns:
            np.ndarray: Updated array of cell densities.
        """
        new_densities = np.copy(self.densities)
        
        # Calculate Sending and Receiving capabilities
        sending = np.zeros(self.num_cells)
        receiving = np.zeros(self.num_cells)
        for i in range(self.num_cells):
            rho = self.densities[i]
            sending[i] = min(self.v_f * rho, self.q_max)
            congested = self.q_max * (1.0 - (rho / self.rho_max))
            receiving[i] = max(0.0, min(self.q_max, congested))
            
        # Calculate flows leaving each cell
        flows_out = np.zeros(self.num_cells + 1)
        
        # Flow into the first cell comes from the observed traffic
        s_input = min(self.v_f * observed_rho_entering, self.q_max)
        flows_out[0] = min(s_input, receiving[0] if self.num_cells > 0 else s_input)
        
        # Flows transferring between cells
        for i in range(self.num_cells - 1):
            flows_out[i+1] = min(sending[i], receiving[i+1])
            
        if self.num_cells > 0:
            flows_out[self.num_cells] = sending[-1]
            
        # Update density in each cell using conservation equation
        # rho(t+1) = rho(t) + (dt / dx) * (q_in - q_out)
        for i in range(self.num_cells):
            q_in = flows_out[i]
            q_out = flows_out[i+1]
            change = (self.dt / self.dx) * (q_in - q_out)
            new_densities[i] += change
            # Prevent negative densities or exceeding jam density due to numerical rounding
            new_densities[i] = max(0.0, min(new_densities[i], self.rho_max))
            
        self.densities = new_densities
        return self.densities
        
    def detect_congestion(self, threshold: float = 0.08) -> list:
        """
        Identifies which cells are currently congested based on threshold.
        Returns a list of boolean values.
        """
        return (self.densities > threshold).tolist()
        
    def plot_fundamental_diagram(self, save_path: str = 'outputs/fundamental_diagram.png'):
        """
        Plots the defined Triangular Fundamental Diagram.
        """
        rhos = np.linspace(0, self.rho_max, 100)
        flows = [self._flow(r) for r in rhos]
        
        plt.figure(figsize=(8,6))
        plt.plot(rhos, flows, label=f"v_f={self.v_f}, q_max={self.q_max}")
        plt.title("Traffic Fundamental Diagram (Flow vs Density)")
        plt.xlabel("Density (vehicles/m)")
        plt.ylabel("Flow (vehicles/s)")
        plt.grid(True)
        plt.legend()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
