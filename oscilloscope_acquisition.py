import pyvisa
import numpy as np
import datetime 

class ScopeAcquisition:
    def __init__(self, scope):
        self.scope = scope
        self.start_time = None
        self.sample_rate = 0.05  # 采样率（秒）
        self._configure_scope()
    
    def _configure_scope(self):
        try:
            # 设置二进制数据格式以加快传输
            self.scope.write("DATA:ENC RPB")
            self.scope.write("DATA:WIDTH 1")
            
            # 设置快速触发模式
            #self.scope.write("TRIGGER:MODE AUTO")
            self.scope.write("TRIGGER:MODE AUTO")  # 使用正常触发模式
            #self.scope.write("TRIGGER:POSITION 20")  # 设置触发位置在20%处
            # self.scope.write("ACQUIRE:MODE SAMPLE")
            # self.scope.write("ACQUIRE:STOPAFTER RUNSTOP")
            
            # 设置较小的记录长度以加快传输
            self.scope.write("HORIZONTAL:RECORDLENGTH 500")
            self.scope.write("HORIZONTAL:POSITION 50") 
            # 设置超时时间
            self.scope.timeout = 2000  # 2秒超时

            
            self.horizontal_scale = float(self.scope.query('HORIZONTAL:SCALE?'))
            #self.sample_rate = 1.0 / (self.horizontal_scale * 2 / 500)  # 计算实际采样率
            
            self.max_point = 500  # 最大采样点数
            self.data_buffer = {ch: [] for ch in range(1, 5)}
            # 计算扫描速度
            total_time_window = self.horizontal_scale * 10  # 示波器显示的总时间窗口（10个格子）
            self.scan_speed = total_time_window / 500  # 每个采样点的时间间隔
            print(f"扫描速度: {self.scan_speed*1000:.3f} ms/点")

            
            # 预先获取波形参数
            self.xze = float(self.scope.query('WFMPRE:XZE?'))
            self.xin = float(self.scope.query('WFMPRE:XIN?'))
            self.ymu = float(self.scope.query('WFMPRE:YMU?'))
            self.yof = float(self.scope.query('WFMPRE:YOF?'))
            
        except Exception as e:
            print(f"Error configuring scope: {e}")
    def get_all_channels_data(self, channels):
        try:
            data_dict = {}
            current_time = datetime.datetime.now()

            if self.start_time is None:
                self.start_time = current_time
            
            elapsed_time = (current_time - self.start_time).total_seconds()
            
            # 使用immediate measurement直接获取电压值
            for ch in channels:
                try:
                    self.scope.write(f"measurement:immed:source1 ch{ch}")
                    self.scope.write("measurement:immed:value?")
                    print(self.scope.read())
                    voltage_value = float(self.scope.read().split("VALUE ")[1])
                    
                    current_voltage = np.array([voltage_value])
                    current_time_point = np.array([elapsed_time])
                    
                    # 仍然获取完整波形用于显示
                    self.scope.write(f"DATA:SOURCE CH{ch}")
                    self.scope.write("CURVE?")
                    data = self.scope.read_raw()
                    
                    if data:
                        header_len = 2 + int(data[1])
                        adc_wave = np.frombuffer(data[header_len:-1], dtype=np.int8)
                        full_voltage = (adc_wave - self.yof) * self.ymu
                        full_time = self.xze + np.arange(len(adc_wave)) * self.xin
                        
                        data_dict[ch] = (
                            (current_time_point, current_voltage),  # 使用immediate measurement值
                            (full_time, full_voltage)
                        )
                        
                except Exception as ch_e:
                    print(f"Error reading channel {ch}: {ch_e}")
                    continue
                    
            return data_dict if data_dict else None
            
        except Exception as e:
            print(f"Error acquiring data: {e}")
            return None

    def _check_scope_ready(self):  # 修复缩进，与其他方法保持一致
        """检查示波器是否就绪"""
        try:
            # 检查示波器是否响应
            idn = self.scope.query("*IDN?")
            if not idn:
                return False
                
            # 清除状态寄存器
            self.scope.write("*CLS")
            
            # 等待操作完成
            self.scope.query("*OPC?")
            
            return True
            
        except Exception as e:
            print(f"Error checking scope status: {e}")
            return False