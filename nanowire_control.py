import numpy as np
from datetime import datetime
from nwm import controller
import time
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
        

     #speed mapping
        self.voltage_mapping_channel = 4  # 默认使用第4通道进行映射
        self.baseline_low = None  # 静止状态基线
        self.baseline_high = None  # 拉伸状态基线
        self.is_collecting_baseline = False
        self.baseline_collection_start = None
        self.baseline_values = {'low': [], 'high': []}
        self.mapping_resolution = 512  # 映射分辨率
        self.min_step = 0.1  # 最小步长 (pixel/s)
        self.max_step = 10.0  # 最大步长 (pixel/s)
        
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
        self.unit_pixel = 20  # 20 pixels per unit distance
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
            
            # 如果PID模式启用，根据电压阈值更新目标位置
            if self.pid_mode:
                self.update_target_by_voltage_threshold(movement_level, direction_level, axis_level)
            else:
                # 非PID模式下，直接应用电压移动
                if axis_level:  # X-axis movement
                    self.move_x(voltage)
                else:  # Y-axis movement
                    self.move_y(voltage)
        else:
            self.current_status['voltage'] = 0.0
            # Apply movement
            if not self.pid_mode:  # 只在非PID模式下应用零电压
                if axis_level:  # X-axis movement
                    self.move_x(0)
                else:  # Y-axis movement
                    self.move_y(0)
            
            # Reset movement time when voltage drops below threshold
            self.last_movement_time = None 
    
    def update_target_by_voltage_threshold(self, movement_level, direction_level, axis_level):
        """根据电压阈值更新目标位置
        
        Args:
            movement_level: 移动通道电压是否超过阈值
            direction_level: 方向通道电压是否超过阈值
            axis_level: 轴选择通道电压是否超过阈值
        """
        # 如果不移动，不更新目标位置
        if not movement_level:
            return
            
        # 设置每次移动的步长（像素）
        step_size = self.unit_pixel
        
        # 根据方向确定步长正负
        if not direction_level:  # 负方向
            step_size = -step_size
            
        # 确保目标位置已初始化
        if self.target_x is None:
            self.target_x = self.current_position['x']
        if self.target_y is None:
            self.target_y = self.current_position['y']
            
        # 根据轴选择更新X或Y目标位置
        if axis_level:  # X轴
            self.target_x += step_size
            # 确保在屏幕范围内（假设屏幕分辨率为1920x1080）
            self.target_x = max(0, min(1920, self.target_x))
        else:  # Y轴
            self.target_y += step_size
            # 确保在屏幕范围内
            self.target_y = max(0, min(1080, self.target_y))
            
        # 应用PID控制移动到新目标位置
        self._apply_pid_control()
        
        # 记录当前时间，用于计算下一次移动
        import time
        self.last_movement_time = time.time()

        # 添加到目标位置历史（用于轨迹显示）
        if not hasattr(self, 'target_history'):
            self.target_history = []
            
        self.target_history.append((time.time(), self.target_x, self.target_y))
        
        # 限制历史记录长度
        max_history = 1000
        if len(self.target_history) > max_history:
            self.target_history = self.target_history[-max_history:]
            
        print(f"Target updated: x={self.target_x}, y={self.target_y}")
    
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
            # 如果是测试模式，使用默认位置
            if self.test_mode:
                self.current_position = {'x': 10, 'y': 10, 'theta': 0.0}
            else:
                # 选择纳米线并获取初始位置
                self._select_nanowire()
                self._update_current_position()
                
            # 初始化目标位置为当前位置
            self.target_x = self.current_position['x']
            self.target_y = self.current_position['y']
            
            print(f"PID mode enabled. Current position: x={self.current_position['x']}, y={self.current_position['y']}")
        else:
            print("PID mode disabled")

    def _update_step_pixel_setting(self,step_size):
        self.unit_pixel = step_size
        print(f'update pixel setting, the step is now: {self.unit_pixel}')
        return True
    
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
        # 测试模式下的处理
        if self.test_mode:
            # 如果在测试模式下，且当前位置已初始化，则直接添加到历史记录
            if self.current_position:
                # 添加到位置历史
                self.position_history.append((
                    time.time(),
                    self.current_position['x'],
                    self.current_position['y']
                ))
                # 限制历史记录长度
                if len(self.position_history) > 1000:
                    self.position_history = self.position_history[-1000:]
                return True
            return False

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
        if self.test_mode:
            # 测试模式下，只更新目标位置，不实际控制硬件
            print(f"[TEST] Moving to target: x={self.target_x}, y={self.target_y}")
            return
            
        if self.selected_nanowire_id is not None:
            try:
                # Set PID setpoints
                controller.set_x_pid_setpoint(self.target_x)
                controller.set_y_pid_setpoint(self.target_y)
                print(f"Moving to target: x={self.target_x}, y={self.target_y}")
            except Exception as e:
                print(f"Error applying PID control: {e}")

### mapping section -------
    def enable_voltage_mapping(self, enabled=True):
        """启用或禁用电压映射功能"""
        self.is_voltage_mapping_enabled = enabled
        print(f"Voltage mapping {'enabled' if enabled else 'disabled'}")
        
    def set_mapping_range(self, min_step, max_step):
        """设置映射步长范围"""
        self.min_step = min_step
        self.max_step = max_step
        print(f"Mapping range set: {min_step} to {max_step} pixel/s")

    def start_baseline_collection(self, state='low'):
        """开始基线采集"""
        self.is_collecting_baseline = True
        self.baseline_collection_start = time.time()
        self.baseline_values[state] = []
        print(f"Started collecting {state} baseline")
        
    def stop_baseline_collection(self, state='low'):
        """停止基线采集并计算平均值"""
        if self.baseline_values[state]:
            if state == 'low':
                self.baseline_low = np.mean(self.baseline_values[state])
            else:
                self.baseline_high = np.mean(self.baseline_values[state])
            print(f"{state} baseline: {self.baseline_low if state == 'low' else self.baseline_high:.3f}V")
        self.is_collecting_baseline = False
        
    def update_voltage_mapping(self, voltage):
        """更新电压映射"""
        if not hasattr(self, 'is_voltage_mapping_enabled') or not self.is_voltage_mapping_enabled:
            return None
            
        if self.is_collecting_baseline:
            current_time = time.time()
            if current_time - self.baseline_collection_start <= 10:  # 10秒采集时间
                if self.baseline_low is None:
                    self.baseline_values['low'].append(voltage)
                else:
                    self.baseline_values['high'].append(voltage)
            else:
                # 自动停止采集
                state = 'low' if self.baseline_low is None else 'high'
                self.stop_baseline_collection(state)
        
        # 如果两个基线都已采集，计算映射步长
        if self.baseline_low is not None and self.baseline_high is not None:
            voltage_range = abs(self.baseline_high - self.baseline_low)
            if voltage_range > 0:
                current_diff = abs(voltage - self.baseline_low)
                # 映射到0-511范围
                mapped_value = int((current_diff / voltage_range) * (self.mapping_resolution - 1))
                # 映射到步长范围
                step_range = self.max_step - self.min_step
                step_size = self.min_step + (step_range * mapped_value / (self.mapping_resolution - 1))
                self.unit_pixel = step_size
                return step_size
        return None