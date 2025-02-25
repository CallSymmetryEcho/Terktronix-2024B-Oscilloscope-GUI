import serial
import numpy as np
import datetime
import time

class ArduinoAcquisition:
    def __init__(self, port='/dev/cu.usbserial-1110', baudrate=115200):
        self.serial = serial.Serial(port, baudrate)
        self.start_time = None
        self.sample_rate = 0.05  # 采样率（秒）
        time.sleep(2)  # 等待Arduino初始化
        self._configure_arduino()
    
    def _configure_arduino(self):
        try:
            # 清空缓冲区
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
        except Exception as e:
            print(f"Error configuring Arduino: {e}")
    
    def get_all_channels_data(self, channels):
        try:
            data_dict = {}
            current_time = datetime.datetime.now()

            if self.start_time is None:
                self.start_time = current_time
            
            elapsed_time = (current_time - self.start_time).total_seconds()
            
            # 读取Arduino发送的数据
            try:
                line = self.serial.readline().decode().strip()
                values = [float(x) for x in line.split(',')[:-1]]  # 最后一个逗号后是空值
                
                for i, ch in enumerate(channels):
                    if i < len(values):
                        # 创建与示波器数据格式相同的结构
                        current_voltage = np.array([values[i]])
                        current_time_point = np.array([elapsed_time])
                        
                        # 创建模拟的波形数据（如果需要显示）
                        sample_count = 500
                        full_time = np.linspace(elapsed_time-0.05, elapsed_time, sample_count)
                        full_voltage = np.ones(sample_count) * values[i]
                        
                        data_dict[ch] = (
                            (current_time_point, current_voltage),
                            (full_time, full_voltage)
                        )
                        
            except Exception as ch_e:
                print(f"Error reading channel {ch}: {ch_e}")
                
            return data_dict if data_dict else None
            
        except Exception as e:
            print(f"Error acquiring data: {e}")
            return None
    
    def _check_arduino_ready(self):
        """检查Arduino是否就绪"""
        try:
            if self.serial.is_open:
                return True
            return False
            
        except Exception as e:
            print(f"Error checking Arduino status: {e}")
            return False
    
    def close(self):
        """关闭串口连接"""
        if hasattr(self, 'serial') and self.serial.is_open:
            self.serial.close()