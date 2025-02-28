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
        
        # PID control parameters
        self.pixel_per_second = 25  # 5 pixels per 0.2 seconds = 25 pixels per second
        self.last_movement_time = None
        self.target_x = None
        self.target_y = None
        self.selected_nanowire_id = None
        self.current_position = {'x': 0, 'y': 0, 'theta': 0}
        self.pid_mode = False  # Flag to enable/disable PID control
        self.position_history = []  # To store position history for trajectory display
    
    def set_channels(self, movement_ch, direction_ch, axis_ch):
        """Set channel numbers for different control signals"""
        self.movement_channel = movement_ch
        self.direction_channel = direction_ch
        self.axis_channel = axis_ch
        print(f"Channels set: Movement={movement_ch}, Direction={direction_ch}, Axis={axis_ch}")
    
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
                
            # If PID mode is enabled, calculate pixel movement based on time
            if self.pid_mode and self.selected_nanowire_id is not None:
                self._update_position_by_voltage_time(movement_level, direction_level, axis_level)
        else:
            self.current_status['voltage'] = 0.0
             # Apply movement
            if axis_level:  # X-axis movement
                self.move_x(0)
            else:  # Y-axis movement
                self.move_y(0)
            
            # Reset movement time when voltage drops below threshold
            self.last_movement_time = None
    
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
    
    # 新增 PID 控制相关方法
    def enable_pid_mode(self, enabled=True):
        """Enable or disable PID control mode"""
        self.pid_mode = enabled
        if enabled:
            # Select nanowire and get initial position
            self._select_nanowire()
            self._update_current_position()
            # Initialize target position to current position
            if self.target_x is None:
                self.target_x = self.current_position['x']
            if self.target_y is None:
                self.target_y = self.current_position['y']
            print(f"PID mode enabled. Current position: x={self.current_position['x']}, y={self.current_position['y']}")
        else:
            print("PID mode disabled")
    
    def _select_nanowire(self):
        """Select nanowire for control"""
        try:
            self.selected_nanowire_id = controller.get_selected_nanowire_id()
            if self.selected_nanowire_id is None:
                print("No nanowire selected. Please select a nanowire first.")
                return False
            return True
        except Exception as e:
            print(f"Error selecting nanowire: {e}")
            return False
    
    def _update_current_position(self):
        """Update current nanowire position"""
        try:
            if self.selected_nanowire_id is None:
                return False
                
            all_nanowires = controller.get_nanowires()
            for nw in all_nanowires:
                if nw.id == self.selected_nanowire_id:
                    self.current_position = {
                        'x': round(float(nw.x)),
                        'y': round(float(nw.y)),
                        'theta': float(nw.theta)
                    }
                    # Add to position history for trajectory display
                    self.position_history.append((
                        time.time(),
                        self.current_position['x'],
                        self.current_position['y']
                    ))
                    # Limit history size
                    if len(self.position_history) > 1000:
                        self.position_history = self.position_history[-1000:]
                    return True
            return False
        except Exception as e:
            print(f"Error updating position: {e}")
            return False
    
    def get_current_position(self):
        """Get current nanowire position"""
        self._update_current_position()
        return self.current_position
    
    def get_position_history(self):
        """Get position history for trajectory display"""
        return self.position_history
    
    def set_target_position(self, x=None, y=None):
        """Set target position for PID control"""
        if x is not None:
            self.target_x = x
        if y is not None:
            self.target_y = y
        print(f"Target position set: x={self.target_x}, y={self.target_y}")
        return True
    
    def get_target_position(self):
        """Get current target position"""
        return {'x': self.target_x, 'y': self.target_y}
    
    def _update_position_by_voltage_time(self, movement_level, direction_level, axis_level):
        """Update target position based on voltage duration"""
        current_time = time.time()
        
        # Initialize last_movement_time if not set
        if self.last_movement_time is None:
            self.last_movement_time = current_time
            return
        
        # Calculate time elapsed since last movement
        elapsed_time = current_time - self.last_movement_time
        
        # Calculate pixel movement based on time
        pixel_movement = int(elapsed_time * self.pixel_per_second)
        
        if pixel_movement > 0:
            # Update target position based on direction and axis
            if axis_level:  # X-axis
                if direction_level:  # Positive direction
                    self.target_x += pixel_movement
                else:  # Negative direction
                    self.target_x -= pixel_movement
            else:  # Y-axis
                if direction_level:  # Positive direction
                    self.target_y += pixel_movement
                else:  # Negative direction
                    self.target_y -= pixel_movement
            
            # Apply PID control to move to target position
            self._apply_pid_control()
            
            # Reset timer
            self.last_movement_time = current_time
    
    def _apply_pid_control(self):
        """Apply PID control to move nanowire to target position"""
        if not self.test_mode and self.selected_nanowire_id is not None:
            try:
                # Set PID setpoints
                controller.set_x_pid_setpoint(self.target_x)
                controller.set_y_pid_setpoint(self.target_y)
                print(f"Moving to target: x={self.target_x}, y={self.target_y}")
            except Exception as e:
                print(f"Error applying PID control: {e}")