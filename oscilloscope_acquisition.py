import pyvisa
import numpy as np
import datetime 
import time

class ScopeAcquisition:
    def __init__(self, scope):
        self.scope = scope
        self.start_time = None
        self.sample_rate = 0.05  # 采样率（秒）
        self._configure_scope()
    
    def _configure_scope(self):
        try:
            # 设置超时时间
            self.scope.timeout = 2000  # 2秒超时
            
            # 设置数据格式
            self.scope.write("DATA:ENC RPB")
            self.scope.write("DATA:WIDTH 1")
            
            # # 设置采集模式
            # self.scope.write("ACQUIRE:MODE SAMPLE")  # 采样模式
            # self.scope.write("ACQUIRE:STOPAFTER SEQUENCE")  # 改为序列模式
            # self.scope.write("ACQUIRE:STATE ON")  # 确保采集开启
            
            # # 设置触发
            # self.scope.write("TRIGGER:MAIN:MODE AUTO")  # 使用主触发自动模式
            # self.scope.write("TRIGGER:MAIN:TYPE EDGE")  # 边沿触发
            # self.scope.write("TRIGGER:MAIN:EDGE:SLOPE RISE")  # 上升沿触发
            
            # # 设置水平时基和记录长度
            # self.scope.write("HORIZONTAL:SCALE 0.05")  # 50ms/div
            # self.scope.write("HORIZONTAL:RECORDLENGTH 500")
            # self.scope.write("HORIZONTAL:POSITION 50")
            
            # # 确保连续运行
            # self.scope.write("ACQUIRE:STATE RUN")
            
            
            # 设置测量源为对应通道
            for ch in range(1, 5):
                self.scope.write(f"measurement:immed:source ch{ch}")
                # 设置测量类型为平均值
                self.scope.write("measurement:immed:type mean")
            

            # self.horizontal_scale = float(self.scope.query('HORIZONTAL:SCALE?'))
            #self.sample_rate = 1.0 / (self.horizontal_scale * 2 / 500)  # 计算实际采样率
            
            self.max_point = 500  # 最大采样点数
            self.data_buffer = {ch: [] for ch in range(1, 5)}

            
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
                    start_total = time.time()
                    
                    # 获取测量值
                    start_meas = time.time()
                    voltage_value = float(self.scope.query("measurement:immed:value?"))
                    print(f"Channel {ch} measurement time: {(time.time()-start_meas)*1000:.2f} ms")
                    
                    current_voltage = np.array([voltage_value])
                    current_time_point = np.array([elapsed_time])
                    
                    # 获取波形数据
                    start_wave = time.time()
                    self.scope.write(f"DATA:SOURCE CH{ch}")
                    self.scope.write("CURVE?")
                    data = self.scope.read_raw()
                    print(f"Channel {ch} waveform acquisition time: {(time.time()-start_wave)*1000:.2f} ms")
                    
                    # 处理数据
                    start_proc = time.time()
                    if data:
                        header_len = 2 + int(data[1])
                        adc_wave = np.frombuffer(data[header_len:-1], dtype=np.int8)
                        full_voltage = (adc_wave - self.yof) * self.ymu
                        full_time = self.xze + np.arange(len(adc_wave)) * self.xin
                        
                        data_dict[ch] = (
                            (current_time_point, current_voltage),
                            (full_time, full_voltage)
                        )
                    print(f"Channel {ch} data processing time: {(time.time()-start_proc)*1000:.2f} ms")
                    print(f"Channel {ch} total time: {(time.time()-start_total)*1000:.2f} ms\n")
                    
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