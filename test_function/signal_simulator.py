import numpy as np
from datetime import datetime
import time

class SignalSimulator:
    def __init__(self, num_channels=4, sample_rate=1000, noise_level=0.1):
        """
        Initialize the signal simulator
        num_channels: Number of channels to simulate (default 4 for 4 fingers)
        sample_rate: Samples per second
        noise_level: Amount of random noise to add (0-1)
        """
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self.time = 0
        self.wave_type = 'Sine'  # 默认波形类型
        self.base_frequencies = [1, 2, 1.5, 2.5]
        self.min_amplitude = -5.0
        self.max_amplitude = 5.0
        
    def set_wave_type(self, wave_type):
        """设置波形类型"""
        self.wave_type = wave_type
        
    def set_amplitude_range(self, min_amplitude, max_amplitude):
        """设置振幅范围"""
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        
    def get_realtime_sample(self):
        """获取实时采样数据"""
        amplitude_range = (self.max_amplitude - self.min_amplitude) / 2
        amplitude_offset = (self.max_amplitude + self.min_amplitude) / 2
        
        values = []
        for i in range(self.num_channels):
            if self.wave_type == 'Sine':
                value = amplitude_range * np.sin(2 * np.pi * self.base_frequencies[i] * self.time) + amplitude_offset
            elif self.wave_type == 'Square':
                value = amplitude_range * np.sign(np.sin(2 * np.pi * self.base_frequencies[i] * self.time)) + amplitude_offset
            elif self.wave_type == 'Sawtooth':
                t = self.time * self.base_frequencies[i]
                value = amplitude_range * (2 * (t - np.floor(t + 0.5))) + amplitude_offset
            
            # 添加噪声
            noise = self.noise_level * np.random.randn()
            value = np.clip(value + noise, self.min_amplitude, self.max_amplitude)
            values.append(value)
        
        self.time += 0.001  # 时间步进
        return values
    
    def reset_time(self):
        """Reset the simulation time"""
        self.time = 0

# Example usage
if __name__ == "__main__":
    # Create simulator instance
    simulator = SignalSimulator()
    
    # Generate a flexion gesture pattern
    t, signals = simulator.generate_gesture_pattern('flex', duration=2.0)
    
    # Print some sample values
    print("Sample values from gesture pattern:")
    for i, signal in enumerate(signals):
        print(f"Channel {i}: {signal[:5]}")
    
    # Test real-time sampling
    print("\nReal-time samples:")
    for _ in range(5):
        values = simulator.get_realtime_sample()
        print(f"Sample: {values}")
        time.sleep(0.1)  # Simulate delay between samples