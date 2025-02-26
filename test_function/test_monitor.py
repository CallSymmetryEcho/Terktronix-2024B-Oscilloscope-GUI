import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import csv
import os
from signal_simulator import SignalSimulator

class TestMonitor:
    def __init__(self, num_channels=4, sample_rate=1000):
        """
        Initialize the test monitor with simulated signals
        num_channels: Number of channels to simulate
        sample_rate: Samples per second
        """
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.running = False
        self.data = [[] for _ in range(num_channels)]
        self.start_time = None
        
        # Initialize signal simulator
        self.simulator = SignalSimulator(num_channels=num_channels, sample_rate=sample_rate)
        
        # Create figure for visualization
        self.setup_plot()
    
    def setup_plot(self):
        """Set up the visualization plot"""
        self.fig, self.axes = plt.subplots(self.num_channels, 1, figsize=(12, 8))
        if self.num_channels == 1:
            self.axes = [self.axes]
        
        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], 'b-', label=f'Channel {i+1}')
            self.lines.append(line)
            ax.set_title(f'Channel {i+1}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Voltage (V)')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        if not self.running:
            return self.lines
        
        # Get new sample from simulator
        values = self.simulator.get_realtime_sample()
        current_time = (datetime.now() - self.start_time).total_seconds()
        
        # Update data arrays
        for i, value in enumerate(values):
            self.data[i].append([current_time, value])
        
        # Update plots
        for i, line in enumerate(self.lines):
            data_array = np.array(self.data[i])
            if len(data_array) > 0:
                line.set_data(data_array[:, 0], data_array[:, 1])
                self.axes[i].relim()
                self.axes[i].autoscale_view()
        
        return self.lines
    
    def save_data(self):
        """Save recorded data to CSV files"""
        if not any(self.data):
            return
        
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save data for each channel
        for i in range(self.num_channels):
            filename = f'data/test_CH{i+1}_{timestamp}.csv'
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time (s)', 'Voltage (V)'])
                writer.writerows(self.data[i])
            
            print(f"Data saved to {filename}")
    
    def run(self, duration=None):
        """Run the test monitor"""
        self.running = True
        self.start_time = datetime.now()
        self.data = [[] for _ in range(self.num_channels)]
        
        # Create animation
        interval = 1000 / self.sample_rate  # Convert to milliseconds
        ani = FuncAnimation(self.fig, self.update_plot, interval=interval, blit=True)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.save_data()
            plt.close()

if __name__ == "__main__":
    # Create and run test monitor
    monitor = TestMonitor(num_channels=4, sample_rate=100)
    monitor.run()