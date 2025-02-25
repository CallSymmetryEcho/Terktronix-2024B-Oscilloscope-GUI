class NanowireTestController:
    def __init__(self):
        self.movement_channel = 1
        self.direction_channel = 2
        self.axis_channel = 3
        
        # Individual voltage thresholds for each control channel
        self.movement_threshold = 0.5
        self.direction_threshold = 0.5
        self.axis_threshold = 0.5
        self.movement_voltage = 1.0
        
        self.current_status = {
            'is_moving': False,
            'direction': '+',
            'axis': 'X',
            'voltage': 0.0
        }
    
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
        else:
            self.current_status['voltage'] = 0.0
    
    def move_x(self, voltage):
        """Simulate X-axis movement"""
        # Just update status, no actual hardware communication
        pass
    
    def move_y(self, voltage):
        """Simulate Y-axis movement"""
        # Just update status, no actual hardware communication
        pass
    
    def get_status(self):
        """Get current control status"""
        return self.current_status