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
        self.is_running = False
    
    def _configure_arduino(self):
        try:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            # 设置较小的读取超时
            self.serial.timeout = 0.2  # 100ms超时
            # 设置较大的写入缓冲区
            self.serial.write_buffer_size = 1024
        except Exception as e:
            print(f"Error configuring Arduino: {e}")
    
    def start_acquisition(self):
        """开始数据采集"""
        try:
            self.serial.write(b'start\n')
            response = self.serial.readline().decode().strip()
            if "Recording started" in response:
                self.is_running = True
                self.start_time = datetime.datetime.now()
                return True
            return False
        except Exception as e:
            print(f"Error starting acquisition: {e}")
            return False
    
    def stop_acquisition(self):
        """停止数据采集"""
        try:
            self.serial.write(b'stop\n')
            response = self.serial.readline().decode().strip()
            if "Recording stopped" in response:
                self.is_running = False
                return True
            return False
        except Exception as e:
            print(f"Error stopping acquisition: {e}")
            return False

    def get_all_channels_data(self, channels):
        try:
            if not self.is_running:
                return None
                
            data_dict = {}
            current_time = datetime.datetime.now()

            if self.start_time is None:
                self.start_time = current_time
            
            # 清空多余数据
            if self.serial.in_waiting > 1000:
                self.serial.reset_input_buffer()
            
            elapsed_time = (current_time - self.start_time).total_seconds()
            
            try:
                line = self.serial.readline().decode().strip()
                if not line:  # 如果是空行则跳过
                    return None
                    
                values = [float(x) for x in line.split(',')[:-1]]
                
                for i, ch in enumerate(channels):
                    if i < len(values):
                        current_voltage = np.array([values[i]])
                        current_time_point = np.array([elapsed_time])
                        
                        sample_count = 500
                        full_time = np.linspace(elapsed_time-0.05, elapsed_time, sample_count)
                        full_voltage = np.ones(sample_count) * values[i]
                        
                        data_dict[ch] = (
                            (current_time_point, current_voltage),
                            (full_time, full_voltage)
                        )
                        
            except Exception as ch_e:
                print(f"Error reading channel data: {ch_e}")
                return None
                
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