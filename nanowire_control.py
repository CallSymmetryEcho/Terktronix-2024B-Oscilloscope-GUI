import numpy as np
from datetime import datetime
from nwm import controller

class NanowireController:
    def __init__(self):
        self.movement_channel = 1  # Default channel for movement control
        self.direction_channel = 2  # Default channel for direction control (+/-)
        self.axis_channel = 3  # Default channel for axis selection (X/Y)
        
        # Individual voltage thresholds for each control channel
        self.movement_threshold = 0.5
        self.direction_threshold = 0.5
        self.axis_threshold = 0.5
        self.movement_voltage = 1.0  # Voltage to apply when moving
        
        self.current_status = {
            'is_moving': False,
            'direction': '+',  # '+' or '-'
            'axis': 'X',      # 'X' or 'Y'
            'voltage': 0.0
        }

        # Test mode support
        self.test_mode = False
        self.test_controller = None
    
    def set_channels(self, movement_ch, direction_ch, axis_ch):
        """Set channel numbers for different control signals"""
        self.movement_channel = movement_ch
        self.direction_channel = direction_ch
        self.axis_channel = axis_ch
    
    def set_voltage_threshold(self, threshold, channel_type='all'):
        """Set the voltage threshold for high/low level detection
        
        Args:
            threshold: The voltage threshold value
            channel_type: 'movement', 'direction', 'axis', or 'all'
        """
        if channel_type == 'movement' or channel_type == 'all':
            self.movement_threshold = threshold
        if channel_type == 'direction' or channel_type == 'all':
            self.direction_threshold = threshold
        if channel_type == 'axis' or channel_type == 'all':
            self.axis_threshold = threshold
    
    def set_movement_voltage(self, voltage):
        """Set the voltage to apply when moving"""
        self.movement_voltage = voltage

    def set_test_mode(self, enabled):
        """Enable or disable test mode"""
        from nanowire_test_control import NanowireTestController
        self.test_mode = enabled
        if enabled:
            self.test_controller = NanowireTestController()
            # Sync current settings to test controller
            self.test_controller.set_channels(self.movement_channel, self.direction_channel, self.axis_channel)
            self.test_controller.set_voltage_threshold(self.movement_threshold, 'movement')
            self.test_controller.set_voltage_threshold(self.direction_threshold, 'direction')
            self.test_controller.set_voltage_threshold(self.axis_threshold, 'axis')
            self.test_controller.set_movement_voltage(self.movement_voltage)
        else:
            self.test_controller = None
    
    def update_control_signals(self, voltages):
        """Update control status based on input voltages from channels"""
        if self.test_mode and self.test_controller:
            self.test_controller.update_control_signals(voltages)
            self.current_status = self.test_controller.get_status()
            return

        # Get voltage levels from control channels
        movement_level = voltages.get(self.movement_channel, 0) > self.movement_threshold
        direction_level = voltages.get(self.direction_channel, 0) > self.direction_threshold
        axis_level = voltages.get(self.axis_channel, 0) > self.axis_threshold
        
        # Update current status
        self.current_status['is_moving'] = movement_level
        self.current_status['direction'] = '+' if direction_level else '-'
        self.current_status['axis'] = 'X' if axis_level else 'Y'
        
        # Calculate movement voltage
        if movement_level:
            voltage = self.movement_voltage if direction_level else -self.movement_voltage
            self.current_status['voltage'] = voltage
            
            # Apply movement
            if axis_level:  # X-axis movement
                self.move_x(voltage)
            else:  # Y-axis movement
                self.move_y(voltage)
        else:
            self.current_status['voltage'] = 0.0
    
    def move_x(self, voltage):
        """Move nanowire in X direction"""
        if not self.test_mode:
            controller.set_x_bias(voltage)
    
    def move_y(self, voltage):
        """Move nanowire in Y direction"""
        if not self.test_mode:
            controller.set_y_bias(voltage)
    
    def get_status(self):
        """Get current control status"""
        return self.current_status