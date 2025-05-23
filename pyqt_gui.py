import sys
import pyvisa
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QCheckBox, QGroupBox, 
                           QLabel, QSpinBox, QFileDialog, QProgressBar, QDialog,
                           QDialogButtonBox, QFormLayout, QInputDialog, QComboBox,
                           QDoubleSpinBox, QGridLayout,QScrollArea,QTabWidget)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, pyqtSlot
import pyqtgraph as pg
from datetime import datetime
import csv
import os
import time
from test_function.signal_simulator import SignalSimulator
from nanowire_control import NanowireController
from oscilloscope_acquisition import ScopeAcquisition
from arduino_reader import ArduinoAcquisition

# -------------------- 保存配置对话框 --------------------
class SaveConfigDialog(QDialog):
    def __init__(self, channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Channels to Save")
        self.setMinimumSize(300, 200)

        
        layout = QVBoxLayout()
        self.checkboxes = {}
        
        form = QFormLayout()
        for ch in channels:
            cb = QCheckBox(f"Channel {ch}")
            cb.setChecked(True)  # 默认全选
            self.checkboxes[ch] = cb
            form.addRow(QLabel(f"CH{ch}:"), cb)
        
        layout.addLayout(form)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def selected_channels(self):
        return [ch for ch, cb in self.checkboxes.items() if cb.isChecked()]

# -------------------- 数据采集线程 --------------------
class AcquisitionThread(QThread):
    data_ready = pyqtSignal(dict)
    
    def __init__(self, scope, active_channels):
        super().__init__()
        self.scope_acquisition = ScopeAcquisition(scope)
        self.active_channels = active_channels
        self.running = True
        
    def run(self):
        while self.running:
            try:
                current_time = datetime.now().timestamp()
                
                # 同步获取所有通道数据
                data = self.scope_acquisition.get_all_channels_data(self.active_channels)
                if data:
                    # data_dict = {
                    #     ch: (current_time, time, voltage) 
                    #     for ch, (time, voltage) in data.items()
                    # }
                    self.data_ready.emit(data)
                
                QThread.msleep(20)  # 控制采集频率
                
            except Exception as e:
                print(f"Acquisition error: {e}")

class TestAcquisitionThread(QThread):
    data_ready = pyqtSignal(dict)
    
    def __init__(self, active_channels, wave_type, min_amplitude, max_amplitude, frequency):
        super().__init__()
        self.active_channels = active_channels
        self.running = True
        self.simulator = SignalSimulator(num_channels=len(active_channels))
        self.simulator.set_wave_type(wave_type)
        self.simulator.set_amplitude_range(min_amplitude, max_amplitude)
        # 设置基础频率
        self.simulator.base_frequencies = [frequency] * len(active_channels)
        self.start_time = None
    def run(self):
        while self.running:
            try:
                data_dict = {}
                current_time = datetime.now().timestamp()
                if self.start_time is None:
                    self.start_time = current_time
                
                # 获取模拟数据
                values = self.simulator.get_realtime_sample()
                
                # 为每个通道创建数据
                for i, ch in enumerate(self.active_channels):
                    time = np.linspace(0, 0.1, 1000)  # 100ms 的时间窗口
                    voltage = np.full_like(time, values[i])  # 使用当前采样值
                    #data_dict[ch] = (current_time, time, voltage)
                    time_pass = current_time - self.start_time
                    #print(time_pass)## debug
                    current_time_point = np.array([time_pass])
                    current_voltage = np.array([values[i]])
                    data_dict[ch] = (
                            (current_time_point, current_voltage),  # 实时电压显示使用同样的采样点
                            (time, voltage)
                        )
                   
                
                if data_dict:
                    self.data_ready.emit(data_dict)
                
                QThread.msleep(50)  # 20Hz 刷新率
                
            except Exception as e:
                print(f"Test acquisition error: {e}")
    
    def _get_waveform(self, channel):
        try:
            # 配置数据源并获取波形参数
            self.scope.write(f"DATA:SOURCE CH{channel}")
            xze = float(self.scope.query('WFMPRE:XZE?'))
            xin = float(self.scope.query('WFMPRE:XIN?'))
            ymu = float(self.scope.query('WFMPRE:YMU?'))
            yof = float(self.scope.query('WFMPRE:YOF?'))
            
            # 获取原始波形数据
            self.scope.write("CURVE?")
            data = self.scope.read_raw()
            header_len = 2 + int(data[1])
            adc_wave = np.frombuffer(data[header_len:-1], dtype=np.int8)
            
            # 转换为时间和电压值
            time = xze + np.arange(len(adc_wave)) * xin
            voltage = (adc_wave - yof) * ymu
            return time, voltage
        except Exception as e:
            print(f"Error acquiring CH{channel} waveform: {e}")
            return None, None

# -------------------- Arduino数据采集线程 --------------------
class ArduinoAcquisitionThread(QThread):
    data_ready = pyqtSignal(dict)
    
    def __init__(self, arduino, active_channels):
        super().__init__()
        self.arduino = arduino
        self.active_channels = active_channels
        self.running = True
        
    def run(self):
        # 启动Arduino数据采集
        if not self.arduino.start_acquisition():
            print("Failed to start Arduino acquisition")
            return
            
        while self.running:
            try:
                data = self.arduino.get_all_channels_data(self.active_channels)
                if data:
                    self.data_ready.emit(data)
                QThread.msleep(20)
                
            except Exception as e:
                print(f"Arduino acquisition error: {e}")
                
    def stop(self):
        self.running = False
        self.arduino.stop_acquisition()

# -------------------- 数据保存线程 --------------------
class SaveThread(QThread):
    finished = pyqtSignal(bool)  # 添加完成信号
    
    def __init__(self, data, scope, channels, save_dir):  # 添加 save_dir 参数
        super().__init__()
        self.data = data
        self.scope = scope
        self.channels = channels
        self.save_dir = save_dir  # 保存目录作为参数传入
        
    def run(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存每个选中通道的数据
            for ch in self.channels:
                if ch in self.data and len(self.data[ch]) > 0:
                    # 保存平均值数据
                    self._save_average_data(ch, self.save_dir, timestamp)
                    
                    # 保存最后波形
                    self._save_last_waveform(ch, self.save_dir, timestamp)
            
            self.finished.emit(True)  # 发送成功信号
        except Exception as e:
            print(f"Save error: {e}")
            self.finished.emit(False)  # 发送失败信号

    def _save_average_data(self, ch, save_dir, timestamp):
        filename = os.path.join(save_dir, f'CH{ch}_average_{timestamp}.csv')
        data_array = np.array(self.data[ch])
        np.savetxt(filename, data_array, 
                  delimiter=',',
                  header='Time(s),Voltage(V)',
                  comments='',
                  fmt=['%.3f', '%.6f'])
    
    def _save_last_waveform(self, ch, save_dir, timestamp):
        try:
            self.scope.write(f"DATA:SOURCE CH{ch}")
            xze = float(self.scope.query('WFMPRE:XZE?'))
            xin = float(self.scope.query('WFMPRE:XIN?'))
            ymu = float(self.scope.query('WFMPRE:YMU?'))
            yof = float(self.scope.query('WFMPRE:YOF?'))
            
            self.scope.write("CURVE?")
            data = self.scope.read_raw()
            header_len = 2 + int(data[1])
            adc_wave = np.frombuffer(data[header_len:-1], dtype=np.int8)
            
            time = xze + np.arange(len(adc_wave)) * xin
            voltage = (adc_wave - yof) * ymu
            
            filename = os.path.join(save_dir, f'CH{ch}_waveform_{timestamp}.csv')
            np.savetxt(filename, np.column_stack((time, voltage)),
                      delimiter=',',
                      header='Time(s),Voltage(V)',
                      comments='',
                      fmt=['%.6f', '%.4f'])
        except Exception as e:
            print(f"Error saving CH{ch} waveform: {e}")

# -------------------- 主界面 --------------------
class OscilloscopeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Multi-Channel Oscilloscope Monitor')
        self.setMinimumSize(1200, 800)
        self.max_data_points = 1000 # 最大数据点数

        # 添加数据源管理
        self.data_source = None  # 'scope' 或 'arduino'
        self.arduino = None      
        
        # 硬件资源管理
        self.rm = pyvisa.ResourceManager()
        self.scope = None
        
        # 数据管理
        self.channels = [1, 2, 3, 4]
        self.active_channels = []
        self.data = {}
        self.save_channels = []
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        
        # 数据收集状态
        self.collecting = False
        self.collection_data = {}
        
        # 线程控制
        self.running = False
        self.acquisition_thread = None
        
        # 纳米线控制器
        self.nanowire_controller = NanowireController()


        # 轨迹初始化
        self.target_trajectory = []
        
        # 初始化界面
        self._init_ui()
        
    def _init_ui(self):
        # 主窗口布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板（使用标签页组织）
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 右侧图表区域（保持不变）
        plot_panel = self._create_plot_panel()
        main_layout.addWidget(plot_panel)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _create_standard_group(self, title, layout_type=QVBoxLayout, max_height=None):
        """Helper method to create a standard group box with layout"""
        group = QGroupBox(title)
        layout = layout_type()
        group.setLayout(layout)
        if max_height:
            group.setMaximumHeight(max_height)
        return group, layout

    def _create_labeled_spinbox(self, label_text, min_val, max_val, default_val, step=0.5, double=False):
        """Helper method to create   labeled spinbox"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        spin = QDoubleSpinBox() if double else QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setSingleStep(step)
        layout.addWidget(spin)
        return layout, spin

    


        self.active_channels = [ch for ch, cb in self.channel_checkboxes.items() if cb.isChecked()]
        
        # 更新图表可见性
        for ch in self.channels:
            visible = ch in self.active_channels
            self.subplots[ch].setVisible(visible)
            self.history_plots[ch].setVisible(visible)

    def _start_acquisition(self):
        if not self.active_channels:
            self.status_bar.showMessage("Please select at least one channel!")
            return
        
        self.running = True
        self.start_time = datetime.now().timestamp()
        self.data = {ch: [] for ch in self.active_channels}
        
        # 根据模式选择采集线程
        if self.test_mode_cb.isChecked():
            # 测试模式使用模拟信号
            self.acquisition_thread = TestAcquisitionThread(
                self.active_channels,
                self.wave_type.currentText(),
                self.min_amplitude_spin.value(),  # 修正：使用正确的振幅范围控件
                self.max_amplitude_spin.value(),
                self.frequency_spin.value()
            )
        else:
            # 实际设备模式
            self.acquisition_thread = AcquisitionThread(self.scope, self.active_channels)
        
        self.acquisition_thread.data_ready.connect(self._update_plots)
        self.acquisition_thread.start()
        
        # 更新按钮状态
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_collect_start.setEnabled(True)  # Enable data collection when acquisition starts
        self.btn_collect_stop.setEnabled(False)
        
        # 启动定时器用于界面刷新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._force_redraw)
        self.update_timer.start(self.interval_spin.value())

    def _stop_acquisition(self):
        self.running = False
        if self.acquisition_thread:
            self.acquisition_thread.running = False
            self.acquisition_thread.quit()
            self.acquisition_thread.wait()
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        self.update_timer.stop()
        self.status_bar.showMessage("Acquisition stopped")

    def _show_save_dialog(self):
        dialog = SaveConfigDialog(self.active_channels)
        if dialog.exec_() == QDialog.Accepted:
            self.save_channels = dialog.selected_channels()
            self._save_data()

    def _save_data(self):
        if not self.save_channels:
            return
        
        # 在主线程中获取保存目录
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
        
        # 创建并启动保存线程
        self.save_thread = SaveThread(self.data, self.scope, self.save_channels, save_dir)
        self.save_thread.finished.connect(self._on_save_completed)
        self.save_thread.start()
        
        # 禁用保存按钮，防止重复操作
        self.btn_save.setEnabled(False)
        self.status_bar.showMessage("Saving selected data in background...")
    
    def _on_save_completed(self, success):
        # 保存完成后的处理
        self.btn_save.setEnabled(True)
        if success:
            self.status_bar.showMessage("Data saved successfully!")
        else:
            self.status_bar.showMessage("Error occurred while saving data!")

    def _force_redraw(self):
        """ 强制刷新图表显示 """
        for ch in self.active_channels:
            self.subplots[ch].enableAutoRange()
            self.subplots[ch].replot()

    
  

    def _update_test_mode(self, state):
        # Enable/disable test mode controls
        is_test_mode = state == Qt.Checked
        self.wave_type.setEnabled(is_test_mode)
        self.min_amplitude_spin.setEnabled(is_test_mode)
        self.max_amplitude_spin.setEnabled(is_test_mode)
        self.frequency_spin.setEnabled(is_test_mode)
        self.btn_test_trigger.setEnabled(is_test_mode)

    def _trigger_test_signal(self):
        # Generate test voltages for nanowire control
        test_voltages = {}
        for ch in range(1, 5):
            if ch == self.movement_ch_spin.value():
                test_voltages[ch] = self.movement_voltage_spin.value()
            elif ch == self.direction_ch_spin.value():
                test_voltages[ch] = self.threshold_spin.value() + 0.1  # Ensure above threshold
            elif ch == self.axis_ch_spin.value():
                test_voltages[ch] = self.threshold_spin.value() + 0.1  # Ensure above threshold
            else:
                test_voltages[ch] = 0.0

        # Update nanowire status with test voltages
        self._update_nanowire_status(test_voltages)

    def _start_data_collection(self):
        """开始数据收集"""
        if not self.active_channels:
            self.status_bar.showMessage("Please select at least one channel!")
            return
        
        self.collecting = True
        self.collection_data = {ch: [] for ch in self.active_channels}
        self.btn_collect_start.setEnabled(False)
        self.btn_collect_stop.setEnabled(True)
        self.status_bar.showMessage("Data collection started...")
    
    def _stop_data_collection(self):
        """停止数据收集并保存数据"""
        if not self.collecting:
            return
            
        self.collecting = False
        self.btn_collect_start.setEnabled(True)
        self.btn_collect_stop.setEnabled(False)
        
        # 弹出标注对话框
        label, ok = QInputDialog.getText(self, 'Data Label', 'Enter label for the collected data:')
        if ok and label:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 在data_dir下创建标注子目录
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            
            # 保存每个通道的数据到标注子目录
            for ch in self.active_channels:
                if ch in self.collection_data and self.collection_data[ch]:
                    filename = os.path.join(label_dir, f'CH{ch}_{timestamp}.csv')
                    data_array = np.array(self.collection_data[ch])
                    np.savetxt(filename, data_array,
                              delimiter=',',
                              header='Time(s),Voltage(V)',
                              comments='',
                              fmt=['%.3f', '%.6f'])
            
            self.status_bar.showMessage(f"Data saved in {label_dir}")
        else:
            self.status_bar.showMessage("Data collection cancelled")
        
        self.collection_data = {}
    
    def _create_standard_group(self, title, layout_type=QVBoxLayout, max_height=None):
        """Helper method to create a standard group box with layout"""
        group = QGroupBox(title)
        layout = layout_type()
        group.setLayout(layout)
        if max_height:
            group.setMaximumHeight(max_height)
        return group, layout

    def _create_labeled_spinbox(self, label_text, min_val, max_val, default_val, step=0.5, double=False):
        """Helper method to create a labeled spinbox"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        spin = QDoubleSpinBox() if double else QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setSingleStep(step)
        layout.addWidget(spin)
        return layout, spin

    def _create_control_panel(self):
        # Create main panel
        panel = QGroupBox("Control Panel")
        panel.setFixedWidth(500)
        

        # create tab layout
        tab_widget = QTabWidget()

         # 创建信号采集标签页
        acquisition_tab = self._create_acquisition_tab()
        tab_widget.addTab(acquisition_tab, "signal acquisition")
        
        # 创建纳米线控制标签页
        nanowire_tab = self._create_nanowire_control_tab()
        tab_widget.addTab(nanowire_tab, "NW control")

        # # Create scroll area
        # scroll = QScrollArea()
        # scroll.setWidgetResizable(True)
        # scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # # Create container widget for scroll area
        # container = QWidget()
        # layout = QVBoxLayout()
        # layout.setSpacing(10)
        # layout.setContentsMargins(10, 10, 10, 10)

            # 创建主面板布局并添加标签页组件
        panel_layout = QVBoxLayout()
        panel_layout.addWidget(tab_widget)
        panel.setLayout(panel_layout)

        self._update_nanowire_test_mode(Qt.Checked) 
    

        return panel

    def _create_acquisition_tab(self):
        # 创建信号采集标签页内容
        tab = QWidget()
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建容器组件
        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
    
        # 数据存储设置
        storage_group, storage_layout = self._create_standard_group("Storage Settings", max_height=120)
        
        # 显示当前存储路径
        self.path_label = QLabel(f"Current path: {self.data_dir}")
        self.path_label.setWordWrap(True)
        storage_layout.addWidget(self.path_label)
        
        # 更改存储路径按钮
        btn_change_path = QPushButton("Change Storage Path")
        btn_change_path.clicked.connect(self._change_storage_path)
        storage_layout.addWidget(btn_change_path)
        
        layout.addWidget(storage_group)

        # 测试模式设置
        test_group, test_layout = self._create_standard_group("Test Mode", max_height=300)
        
        # 测试模式开关
        self.test_mode_cb = QCheckBox("Enable Test Mode")
        test_layout.addWidget(self.test_mode_cb)
        
        # 波形选择
        wave_layout = QHBoxLayout()
        wave_layout.addWidget(QLabel("Waveform:"))
        self.wave_type = QComboBox()
        self.wave_type.addItems(["Sine", "Square", "Sawtooth"])
        wave_layout.addWidget(self.wave_type)
        test_layout.addLayout(wave_layout)
        
        # 振幅设置
        min_amp_layout, self.min_amplitude_spin = self._create_labeled_spinbox("Min:", -10.0, 0.0, -5.0, 0.1, True)
        max_amp_layout, self.max_amplitude_spin = self._create_labeled_spinbox("Max:", 0.0, 10.0, 5.0, 0.1, True)
        
        amp_layout = QVBoxLayout()
        amp_layout.addWidget(QLabel("Amplitude Range (V):"))
        amp_layout.addLayout(min_amp_layout)
        amp_layout.addLayout(max_amp_layout)
        test_layout.addLayout(amp_layout)
        
        # 频率设置
        freq_layout, self.frequency_spin = self._create_labeled_spinbox("Frequency (Hz):", 0.1, 150.0, 20.0, 0.1, True)
        test_layout.addLayout(freq_layout)
        
        # 添加触发按钮
        self.btn_test_trigger = QPushButton("Trigger Test Signal")
        self.btn_test_trigger.setEnabled(False)
        self.btn_test_trigger.clicked.connect(self._trigger_test_signal)
        test_layout.addWidget(self.btn_test_trigger)
        
        # 连接测试模式开关信号
        self.test_mode_cb.stateChanged.connect(self._update_test_mode)
        
        layout.addWidget(test_group)
        
        # 通道选择
        channel_group, channel_layout = self._create_standard_group("Active Channels", max_height=180)
        self.channel_checkboxes = {}
        for ch in self.channels:
            cb = QCheckBox(f"Channel {ch}")
            cb.stateChanged.connect(self._update_active_channels)
            self.channel_checkboxes[ch] = cb
            channel_layout.addWidget(cb)
        layout.addWidget(channel_group)
        
        # 采集设置
        settings_group, settings_layout = self._create_standard_group("Acquisition Settings", max_height=100)
        interval_layout, self.interval_spin = self._create_labeled_spinbox("Update Interval (ms):", 10, 5000, 200, 50)
        settings_layout.addLayout(interval_layout)
        layout.addWidget(settings_group)
        
        # 实时电压显示
        voltage_group, voltage_layout = self._create_standard_group("Real-time Voltage", max_height=150)
        self.voltage_labels = {}
        for ch in self.channels:
            label = QLabel(f"CH{ch}: 0.000 V")
            self.voltage_labels[ch] = label
            voltage_layout.addWidget(label)
        layout.addWidget(voltage_group)

        #不同的采集设备
        source_group, source_layout = self._create_standard_group("Data Source", max_height=120)
        
        # 添加数据源选择下拉框
        source_layout.addWidget(QLabel("Select Data Source:"))
        self.source_selector = QComboBox()
        self.source_selector.addItems(["Oscilloscope", "Arduino"])
        self.source_selector.currentTextChanged.connect(self._change_data_source)
        source_layout.addWidget(self.source_selector)
        
        # 添加 Arduino 端口设置
        self.port_selector = QComboBox()
        self.port_selector.setEnabled(False)
        self.btn_refresh_ports = QPushButton("Refresh Ports")
        self.btn_refresh_ports.clicked.connect(self._refresh_serial_ports)
        self.btn_refresh_ports.setEnabled(False)
        
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Arduino Port:"))
        port_layout.addWidget(self.port_selector)
        port_layout.addWidget(self.btn_refresh_ports)
        source_layout.addLayout(port_layout)
        
        layout.addWidget(source_group)
        
        # 控制按钮
        self.btn_connect = QPushButton("Connect Device")
        self.btn_connect.clicked.connect(self._connect_scope)
        
        self.btn_start = QPushButton("Start Acquisition")
        self.btn_start.clicked.connect(self._start_acquisition)
        self.btn_start.setEnabled(False)
        
        self.btn_stop = QPushButton("Stop Acquisition")
        self.btn_stop.clicked.connect(self._stop_acquisition)
        self.btn_stop.setEnabled(False)
        
        self.btn_save = QPushButton("Save Selected Data")
        self.btn_save.clicked.connect(self._show_save_dialog)
        self.btn_save.setEnabled(False)
        
        # 数据收集按钮
        self.btn_collect_start = QPushButton("Start Collection")
        self.btn_collect_start.clicked.connect(self._start_data_collection)
        self.btn_collect_start.setEnabled(False)
        
        self.btn_collect_stop = QPushButton("Stop Collection")
        self.btn_collect_stop.clicked.connect(self._stop_data_collection)
        self.btn_collect_stop.setEnabled(False)
        
        # 数据收集组
        collection_group = QGroupBox("Data Collection")
        collection_layout = QVBoxLayout()
        collection_layout.addWidget(self.btn_collect_start)
        collection_layout.addWidget(self.btn_collect_stop)
        collection_group.setLayout(collection_layout)

        # 添加按钮到布局
        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.btn_connect)
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_stop)
        buttons_layout.addWidget(self.btn_save)
        
        layout.addLayout(buttons_layout)
        layout.addWidget(collection_group)
        
        # 设置容器布局
        container.setLayout(layout)
        
        # 将容器添加到滚动区域
        scroll.setWidget(container)
        
        # 创建标签页布局
        tab_layout = QVBoxLayout()
        tab_layout.addWidget(scroll)
        tab.setLayout(tab_layout)

        return tab


    def _create_nanowire_control_tab(self):
        # 创建纳米线控制标签页内容
        tab = QWidget()
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 修改为需要时显示水平滚动条
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建容器组件
        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # 纳米线控制组
        nanowire_group = QGroupBox("Nanowire Control")
        nanowire_layout = QVBoxLayout()
        nanowire_layout.setSpacing(15)  # 增加间距
        
        # Test Mode Toggle
        test_mode_layout = QHBoxLayout()
        test_mode_layout.addWidget(QLabel("Test Mode:"))
        self.nanowire_test_mode_cb = QCheckBox()
        self.nanowire_test_mode_cb.setChecked(True)
        self.nanowire_test_mode_cb.stateChanged.connect(self._update_nanowire_test_mode)
        test_mode_layout.addWidget(self.nanowire_test_mode_cb)
        test_mode_layout.addStretch()
        nanowire_layout.addLayout(test_mode_layout)
        
        # PID Mode Toggle
        pid_mode_layout = QHBoxLayout()
        pid_mode_layout.addWidget(QLabel("PID Control Mode:"))
        self.pid_mode_cb = QCheckBox()
        self.pid_mode_cb.setChecked(False)
        self.pid_mode_cb.stateChanged.connect(self._update_pid_mode)
        pid_mode_layout.addWidget(self.pid_mode_cb)
        pid_mode_layout.addStretch()
        nanowire_layout.addLayout(pid_mode_layout)

        # Channel Settings
        channel_settings = QGroupBox("Channel Settings")
        channel_settings_layout = QVBoxLayout()
        channel_settings.setMaximumHeight(200)  # 限制高度
        
        # Movement Channel
        movement_ch_layout, self.movement_ch_spin = self._create_labeled_spinbox("Movement Channel:", 1, 4, 1)
        self.movement_ch_spin.valueChanged.connect(self._update_nanowire_channels)
        channel_settings_layout.addLayout(movement_ch_layout)
        
        # Direction Channel
        direction_ch_layout, self.direction_ch_spin = self._create_labeled_spinbox("Direction Channel:", 1, 4, 2)
        self.direction_ch_spin.valueChanged.connect(self._update_nanowire_channels)
        channel_settings_layout.addLayout(direction_ch_layout)
        
        # Axis Channel
        axis_ch_layout, self.axis_ch_spin = self._create_labeled_spinbox("Axis Channel:", 1, 4, 3)
        self.axis_ch_spin.valueChanged.connect(self._update_nanowire_channels)
        channel_settings_layout.addLayout(axis_ch_layout)
        
        # 添加更新按钮
        update_channels_btn = QPushButton("Update Channels Settings")
        update_channels_btn.clicked.connect(self._force_update_channels)
        channel_settings_layout.addWidget(update_channels_btn) 
        
        channel_settings.setLayout(channel_settings_layout)
        nanowire_layout.addWidget(channel_settings)
        
        # Voltage Settings
        voltage_settings = QGroupBox("Voltage Settings")
        voltage_settings_layout = QVBoxLayout()
        voltage_settings.setMaximumHeight(200)  # 限制高度

         # 添加Voltage Mapping设置
        voltage_mapping = QGroupBox("Voltage Mapping")
        voltage_mapping_layout = QVBoxLayout()
        voltage_mapping.setMaximumHeight(200)
        # 添加启用映射选项
        mapping_enable_layout = QHBoxLayout()
        mapping_enable_layout.addWidget(QLabel("Enable Mapping:"))
        self.mapping_enable_cb = QCheckBox()
        self.mapping_enable_cb.setChecked(False)
        self.mapping_enable_cb.stateChanged.connect(self._update_mapping_enable)
        mapping_enable_layout.addWidget(self.mapping_enable_cb)
        mapping_enable_layout.addStretch()
        voltage_mapping_layout.addLayout(mapping_enable_layout)

        # Mapping Channel选择
        mapping_ch_layout, self.mapping_ch_spin = self._create_labeled_spinbox("Mapping Channel:", 1, 4, 4)
        voltage_mapping_layout.addLayout(mapping_ch_layout)

        # 添加最小和最大步长设置
        min_step_layout, self.min_step_spin = self._create_labeled_spinbox("Min Step (pixels/s):", 0.1, 10.0, 0.1, 0.1, True)
        voltage_mapping_layout.addLayout(min_step_layout)
        
        max_step_layout, self.max_step_spin = self._create_labeled_spinbox("Max Step (pixels/s):", 0.1, 20.0, 10.0, 0.1, True)
        voltage_mapping_layout.addLayout(max_step_layout)
        
        # 基线采集按钮
        baseline_layout = QHBoxLayout()
        self.btn_collect_low = QPushButton("Collect Low Baseline")
        self.btn_collect_high = QPushButton("Collect High Baseline")
        baseline_layout.addWidget(self.btn_collect_low)
        baseline_layout.addWidget(self.btn_collect_high)
        voltage_mapping_layout.addLayout(baseline_layout)
        
        # 当前映射状态显示
        self.mapping_status = QLabel("Mapping Status: Not Configured")
        voltage_mapping_layout.addWidget(self.mapping_status)
        
        # 连接信号
        self.btn_collect_low.clicked.connect(lambda: self._start_baseline_collection('low'))
        self.btn_collect_high.clicked.connect(lambda: self._start_baseline_collection('high'))
        self.min_step_spin.valueChanged.connect(self._update_mapping_range)
        self.max_step_spin.valueChanged.connect(self._update_mapping_range)
        voltage_mapping.setLayout(voltage_mapping_layout)
        nanowire_layout.addWidget(voltage_mapping)
        
        # Movement Threshold
        movement_threshold_layout, self.movement_threshold_spin = self._create_labeled_spinbox("Movement Threshold (V):", 0.0, 5.0, 0.5, 0.1, True)
        self.movement_threshold_spin.valueChanged.connect(lambda: self._update_voltage_threshold('movement'))
        voltage_settings_layout.addLayout(movement_threshold_layout)
        
        # Direction Threshold
        direction_threshold_layout, self.direction_threshold_spin = self._create_labeled_spinbox("Direction Threshold (V):", 0.0, 5.0, 0.5, 0.1, True)
        self.direction_threshold_spin.valueChanged.connect(lambda: self._update_voltage_threshold('direction'))
        voltage_settings_layout.addLayout(direction_threshold_layout)
        
        # Axis Threshold
        axis_threshold_layout, self.axis_threshold_spin = self._create_labeled_spinbox("Axis Threshold (V):", 0.0, 5.0, 0.5, 0.1, True)
        self.axis_threshold_spin.valueChanged.connect(lambda: self._update_voltage_threshold('axis'))
        voltage_settings_layout.addLayout(axis_threshold_layout)
        
        # Movement Voltage
        movement_v_layout, self.movement_voltage_spin = self._create_labeled_spinbox("Movement (V):", 0.0, 10.0, 1.0, 0.1, True)
        self.movement_voltage_spin.valueChanged.connect(self._update_movement_voltage)
        voltage_settings_layout.addLayout(movement_v_layout)

        # 添加移动步长控制
        step_size_layout, self.step_size_spin = self._create_labeled_spinbox("Movement Step (pixels):",0.1, 20.0, 0.1, 0.1, True)
        self.step_size_spin.valueChanged.connect(self._update_movement_step_size)
        voltage_settings_layout.addLayout(step_size_layout)
        
        
        voltage_settings.setLayout(voltage_settings_layout)
        nanowire_layout.addWidget(voltage_settings)

        # Target Position Settings
        target_pos_group = QGroupBox("Target Position")
        target_pos_layout = QGridLayout()
        target_pos_group.setMaximumHeight(150)  # 限制高度
        
        # X Position
        target_pos_layout.addWidget(QLabel("Target X:"), 0, 0)
        self.target_x_spin = QSpinBox()
        self.target_x_spin.setRange(0, 1920)
        self.target_x_spin.setValue(10)
        self.target_x_spin.valueChanged.connect(self._update_target_position)
        target_pos_layout.addWidget(self.target_x_spin, 0, 1)
        
        # Y Position
        target_pos_layout.addWidget(QLabel("Target Y:"), 1, 0)
        self.target_y_spin = QSpinBox()
        self.target_y_spin.setRange(0, 1080)
        self.target_y_spin.setValue(10)
        self.target_y_spin.valueChanged.connect(self._update_target_position)
        target_pos_layout.addWidget(self.target_y_spin, 1, 1)
        
        # Set Target Button
        self.btn_set_target = QPushButton("Set Target")
        self.btn_set_target.clicked.connect(self._set_target_position)
        target_pos_layout.addWidget(self.btn_set_target, 2, 0, 1, 2)
        
        target_pos_group.setLayout(target_pos_layout)
        nanowire_layout.addWidget(target_pos_group)
        
        # Current Position Display
        current_pos_group = QGroupBox("Current Position")
        current_pos_layout = QGridLayout()
        current_pos_group.setMaximumHeight(150)  # 限制高度
        
        # X Position
        current_pos_layout.addWidget(QLabel("Current X:"), 0, 0)
        self.current_x_label = QLabel("0")
        current_pos_layout.addWidget(self.current_x_label, 0, 1)
        
        # Y Position
        current_pos_layout.addWidget(QLabel("Current Y:"), 1, 0)
        self.current_y_label = QLabel("0")
        current_pos_layout.addWidget(self.current_y_label, 1, 1)
    
        # Theta Angle
        current_pos_layout.addWidget(QLabel("Theta:"), 2, 0)
        self.current_theta_label = QLabel("0.0°")
        current_pos_layout.addWidget(self.current_theta_label, 2, 1)
        
        current_pos_group.setLayout(current_pos_layout)
        nanowire_layout.addWidget(current_pos_group)
    
        # Status Display
        status_display = QGroupBox("Linear Control Status")
        status_display.setObjectName("Linear Control Status") 
        status_layout = QGridLayout()
        status_display.setMaximumHeight(180)  # 限制高度
        
        # Movement Status
        status_layout.addWidget(QLabel("Movement:"), 0, 0)
        self.moving_label = QLabel("Not Moving")
        status_layout.addWidget(self.moving_label, 0, 1)
        
        # Direction Status
        status_layout.addWidget(QLabel("Direction:"), 1, 0)
        self.direction_label = QLabel("+")
        status_layout.addWidget(self.direction_label, 1, 1)
        
        # Axis Status
        status_layout.addWidget(QLabel("Axis:"), 2, 0)
        self.axis_label = QLabel("X")
        status_layout.addWidget(self.axis_label, 2, 1)
        
        # Voltage Status
        status_layout.addWidget(QLabel("Voltage:"), 3, 0)
        self.voltage_label = QLabel("0.000 V")
        status_layout.addWidget(self.voltage_label, 3, 1)
        
        status_display.setLayout(status_layout)
        nanowire_layout.addWidget(status_display)

        # Add trajectory visualization panel
        trajectory_group = QGroupBox("Trajectory Visualization")
        trajectory_layout = QVBoxLayout()
        
        # Create trajectory plot - 减小尺寸以适应标签页
        self.trajectory_plot = pg.PlotWidget()
        self.trajectory_plot.setBackground('w')
        self.trajectory_plot.setLabel('left', 'Y Position', 'pixels')
        self.trajectory_plot.setLabel('bottom', 'X Position', 'pixels')
        self.trajectory_plot.setTitle('Nanowire Trajectory')
        self.trajectory_plot.showGrid(x=True, y=True)
        self.trajectory_plot.setMinimumHeight(250)  # 设置最小高度
        self.trajectory_plot.setMinimumWidth(350)   # 设置最小宽度
        
        # 设置轨迹图的范围
        self.trajectory_plot.setXRange(0, 1920)
        self.trajectory_plot.setYRange(0, 1080)
        
        # Create curves for target path (red) and actual path (blue)
        self.target_path = self.trajectory_plot.plot(pen=pg.mkPen('r', width=2), name="Target Path")
        self.actual_path = self.trajectory_plot.plot(pen=pg.mkPen('b', width=2), name="Actual Path")
        self.current_pos_marker = self.trajectory_plot.plot([0], [0], 
                                                        pen=None, 
                                                        symbol='o', 
                                                        symbolSize=10, 
                                                        symbolBrush='g')
        
        # 添加图例
        legend = self.trajectory_plot.addLegend()
        legend.addItem(self.target_path, "Target Path")
        legend.addItem(self.actual_path, "Actual Path")
        
        trajectory_layout.addWidget(self.trajectory_plot)
        
        # Add clear trajectory button
        self.btn_clear_trajectory = QPushButton("Clear Trajectory")
        self.btn_clear_trajectory.clicked.connect(self._clear_trajectory)
        trajectory_layout.addWidget(self.btn_clear_trajectory)
        
        trajectory_group.setLayout(trajectory_layout)
        nanowire_layout.addWidget(trajectory_group)
        
        nanowire_group.setLayout(nanowire_layout)

        # 将纳米线控制组添加到主布局
        layout.addWidget(nanowire_group)
    
        # 设置容器布局
        container.setLayout(layout)
        
        # 将容器添加到滚动区域
        scroll.setWidget(container)
        
        # 创建标签页布局
        tab_layout = QVBoxLayout()
        tab_layout.addWidget(scroll)
        tab.setLayout(tab_layout)
        
        return tab

            
    
        # # Set fixed heights for groups to prevent them from expanding too much
        # storage_group.setMaximumHeight(120)
        # test_group.setMaximumHeight(300)
        # channel_group.setMaximumHeight(180)
        # settings_group.setMaximumHeight(100)
        # voltage_group.setMaximumHeight(150)
        # collection_group.setMaximumHeight(100)
        # nanowire_group.setMaximumHeight(400)

        # # Add all widgets to the layout
        # layout.addWidget(storage_group)
        # layout.addWidget(test_group)
        # layout.addWidget(channel_group)
        # layout.addWidget(settings_group)
        # layout.addWidget(voltage_group)
        # layout.addWidget(self.btn_connect)
        # layout.addWidget(self.btn_start)
        # layout.addWidget(self.btn_stop)
        # layout.addWidget(self.btn_save)
        # layout.addWidget(collection_group)
        # layout.addWidget(nanowire_group)
        # layout.addStretch()

        # # Set the layout to the container
        # container.setLayout(layout)
        
        # # Add the container to the scroll area
        # scroll.setWidget(container)
        
        # # Create the main panel layout
        # panel_layout = QVBoxLayout()
        # panel_layout.addWidget(scroll)
        # panel.setLayout(panel_layout)
        # self._update_nanowire_test_mode(Qt.Checked) 
        # return panel

    def _create_plot_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 创建上下两个图形布局
        self.waveform_layout = pg.GraphicsLayoutWidget()  # 实时波形
        self.history_layout = pg.GraphicsLayoutWidget()   # 历史电压
        
        # 设置最小大小
        self.waveform_layout.setMinimumSize(800, 400)
        self.history_layout.setMinimumSize(800, 300)
        
        # 设置布局间距
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 设置布局比例
        layout.addWidget(self.waveform_layout, stretch=3)  # 上部分占比更大
        layout.addWidget(self.history_layout, stretch=2)   # 下部分占比较小
        
        self.subplots = {}      # 实时波形图
        self.wave_curves = {}   # 实时波形曲线
        self.history_plots = {} # 历史电压图
        self.history_curves = {} # 历史电压曲线
        
        colors = [
            (31, 119, 180),   # 蓝色
            (255, 127, 14),   # 橙色
            (44, 160, 44),    # 绿色
            (214, 39, 40)     # 红色
        ]
        
        # 创建实时波形图
        for i, ch in enumerate(self.channels):
            # 实时波形子图
            plot = self.waveform_layout.addPlot(row=i//2, col=i%2, title=f"CH{ch} Real-time Waveform")
            plot.setLabel('left', 'Voltage', 'V')
            plot.setLabel('bottom', 'Time', 's')
            plot.showAxis('right', False)
            plot.showAxis('top', False)
            plot.setMenuEnabled(False)
            plot.setVisible(False)
            
            # 设置子图大小策略
            plot.getViewBox().setDefaultPadding(0.1)  # 设置边距
            plot.setMinimumWidth(350)  # 设置最小宽度
            plot.setMinimumHeight(200)  # 设置最小高度
            
            pen = pg.mkPen(color=colors[i], width=1.5)
            curve = plot.plot(pen=pen, antialias=True)
            
            self.subplots[ch] = plot
            self.wave_curves[ch] = curve
            
            # 历史电压子图
            hist_plot = self.history_layout.addPlot(row=i//2, col=i%2, title=f"CH{ch} Voltage History")
            hist_plot.setLabel('left', 'Voltage', 'V')
            hist_plot.setLabel('bottom', 'Time', 's')
            hist_plot.showAxis('right', False)
            hist_plot.showAxis('top', False)
            hist_plot.setMenuEnabled(False)
            hist_plot.setVisible(False)
            
            # 设置历史图表大小
            hist_plot.getViewBox().setDefaultPadding(0.1)
            hist_plot.setMinimumWidth(350)
            hist_plot.setMinimumHeight(150)
            
            hist_curve = hist_plot.plot(pen=pen, antialias=True)
            
            self.history_plots[ch] = hist_plot
            self.history_curves[ch] = hist_curve
        
        panel.setLayout(layout)
        return panel
    def _change_data_source(self, source):
            """切换数据源"""
            if source == "Arduino":
                self.btn_connect.setText("Connect Arduino")
                self.port_selector.setEnabled(True)
                self.btn_refresh_ports.setEnabled(True)
                self._refresh_serial_ports()
                self.data_source = "arduino"
            else:
                self.btn_connect.setText("Connect Device")
                self.port_selector.setEnabled(False)
                self.btn_refresh_ports.setEnabled(False)
                self.data_source = "scope"
    
    def _refresh_serial_ports(self):
        """刷新可用串口列表"""
        import serial.tools.list_ports
        
        self.port_selector.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_selector.addItems(ports)
    # -------------------- 槽函数 --------------------
    @pyqtSlot(dict)
    def _update_plots(self, data_dict):
        try:
            control_signals = {}
            
            for ch, data in data_dict.items():
                real_time_data, waveform_data = data  # 解包两种数据
                current_time_point, current_voltage = real_time_data  # 实时点数据
                full_time, full_voltage = waveform_data  # 完整波形数据
                
                current_time = current_time_point[0]
                current_voltage_value = current_voltage[0]

                # 更新纳米线控制器状态
                if ch in [self.movement_ch_spin.value(), 
                            self.direction_ch_spin.value(), 
                            self.axis_ch_spin.value()]:
                    control_signals[ch] = current_voltage_value
                # 添加电压映射处理
                if ch == self.mapping_ch_spin.value():
                    mapped_step = self.nanowire_controller.update_voltage_mapping(current_voltage_value)
                    if mapped_step is not None:
                        self.mapping_status.setText(f"Current Step: {mapped_step:.2f} pixel/s")
                 # 如果正在收集数据，保存到collection_data
                if self.collecting and ch in self.active_channels:
                    if ch not in self.collection_data:
                        self.collection_data[ch] = []
                    self.collection_data[ch].append([current_time, current_voltage_value])
                 # 收集纳米线控制信号
                # print(f"check the type {type(ch)} {ch}")
                # if ch == self.movement_ch_spin.value():
                #     control_signals['movement'] = current_voltage_value
                # elif ch == self.direction_ch_spin.value():
                #     control_signals['direction'] = current_voltage_value
                # elif ch == self.axis_ch_spin.value():
                #     control_signals['axis'] = current_voltage_value 

                if ch in self.active_channels:
                    # 更新实时波形
                    self.subplots[ch].setVisible(True)
                    self.wave_curves[ch].setData(full_time, full_voltage)
                    
                    # 更新历史电压图
                    self.history_plots[ch].setVisible(True)
                    if ch not in self.data:
                        self.data[ch] = []
                    
                    self.data[ch].append([current_time, current_voltage_value])
                    
                    # 限制数据点数量
                    if len(self.data[ch]) > self.max_data_points:
                        self.data[ch] = self.data[ch][-self.max_data_points:]
                    
                    # 如果正在收集数据，保存到collection_data
                    if self.collecting and ch in self.collection_data:
                        self.collection_data[ch].append([current_time, current_voltage_value])
                        if len(self.collection_data[ch]) > self.max_data_points:
                            self.collection_data[ch] = self.collection_data[ch][-self.max_data_points:]
                    
                    # 转换数据格式用于绘图
                    data_array = np.array(self.data[ch])
                    self.history_curves[ch].setData(data_array[:, 0], data_array[:, 1])
                    
                    # 更新实时电压显示
                    self.voltage_labels[ch].setText(f"CH{ch}: {current_voltage_value:.3f} V")
                else:
                    self.subplots[ch].setVisible(False)
                    self.history_plots[ch].setVisible(False)

            # 更新纳米线控制器状态
            if control_signals:
                # 更新控制信号，无论是否在PID模式下
                self.nanowire_controller.update_control_signals(control_signals)
                
                # 如果不在PID模式下，更新状态显示
                if self.pid_mode_cb.isChecked():
                    # 获取最新的目标位置
                    target_pos = self.nanowire_controller.get_target_position()
                    
                    # 更新UI上的目标位置显示
                    self.target_x_spin.blockSignals(True)
                    self.target_y_spin.blockSignals(True)
                    self.target_x_spin.setValue(int(target_pos['x']))
                    self.target_y_spin.setValue(int(target_pos['y']))
                    self.target_x_spin.blockSignals(False)
                    self.target_y_spin.blockSignals(False)
                    
                    # 添加到目标轨迹
                    if not hasattr(self, 'target_trajectory'):
                        self.target_trajectory = []
                    
                    self.target_trajectory.append((time.time(), target_pos['x'], target_pos['y']))
                    
                    # 限制轨迹长度
                    max_trajectory = 1000
                    if len(self.target_trajectory) > max_trajectory:
                        self.target_trajectory = self.target_trajectory[-max_trajectory:]
                else:
                    # 非PID模式下，更新状态显示
                    status = self.nanowire_controller.get_status()
                    self._update_nanowire_status(status)
            
            # 更新进度条
            elapsed = current_time
            self.progress_bar.setValue(int(elapsed % 100))
        except Exception as e:
            print(f"Error updating plots: {e}")
            print(f"Control signals: {control_signals}")


    # -------------------- 业务逻辑 --------------------
    def _connect_scope(self):
        """连接设备（示波器或Arduino）"""
        if self.data_source == "arduino":
            try:
                port = self.port_selector.currentText()
                self.arduino = ArduinoAcquisition(port=port)
                self.status_bar.showMessage(f"Connected to Arduino on {port}")
                self.btn_start.setEnabled(True)
                self.btn_connect.setEnabled(False)
                self.btn_collect_start.setEnabled(True)
            except Exception as e:
                self.status_bar.showMessage(f"Arduino connection error: {str(e)}")
        else:
            # 原有的示波器连接代码
            
            try:
                resources = self.rm.list_resources()
                for resource in resources:
                    if 'USB' in resource:
                        self.scope = self.rm.open_resource(
                            resource,
                            timeout=5000,
                            read_termination='\n',
                            write_termination='\n'
                        )
                        idn = self.scope.query("*IDN?").strip()
                        self.status_bar.showMessage(f"Connected: {idn}")
                        
                        # 配置默认通道参数
                        for ch in self.channels:
                            self.scope.write(f"CH{ch}:COUPLING DC")
                            self.scope.write(f"CH{ch}:SCALE 2.0")
                        
                        self.btn_start.setEnabled(True)
                        self.btn_connect.setEnabled(False)
                        self.btn_collect_start.setEnabled(True)  # Enable data collection after connection
                        return
                self.status_bar.showMessage("No compatible device found!")
            except Exception as e:
                self.status_bar.showMessage(f"Connection error: {str(e)}")

    def _update_active_channels(self):
        self.active_channels = [ch for ch, cb in self.channel_checkboxes.items() if cb.isChecked()]
        
        # 更新图表可见性
        for ch in self.channels:
            visible = ch in self.active_channels
            self.subplots[ch].setVisible(visible)
            self.history_plots[ch].setVisible(visible)

    def _start_acquisition(self):
        if not self.active_channels:
            self.status_bar.showMessage("Please select at least one channel!")
            return
        
        self.running = True
        self.start_time = datetime.now().timestamp()
        self.data = {ch: [] for ch in self.active_channels}
        
        # 根据模式选择采集线程
        if self.test_mode_cb.isChecked():
            # 测试模式使用模拟信号
            self.acquisition_thread = TestAcquisitionThread(
                self.active_channels,
                self.wave_type.currentText(),
                self.min_amplitude_spin.value(),  # 修正：使用正确的振幅范围控件
                self.max_amplitude_spin.value(),
                self.frequency_spin.value()
            )
        elif self.data_source == "arduino":
            # Arduino模式
            self.acquisition_thread = ArduinoAcquisitionThread(self.arduino, self.active_channels)
            self.acquisition_thread.data_ready.connect(self._update_plots)
            self.acquisition_thread.start()
        else:
            # 实际设备模式
            self.acquisition_thread = AcquisitionThread(self.scope, self.active_channels)
        
        self.acquisition_thread.data_ready.connect(self._update_plots)
        self.acquisition_thread.start()
        
        # 更新按钮状态
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_collect_start.setEnabled(True)  # Enable data collection when acquisition starts
        self.btn_collect_stop.setEnabled(False)
        
        # 启动定时器用于界面刷新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._force_redraw)
        self.update_timer.start(self.interval_spin.value())

    def _stop_acquisition(self):
        self.running = False
        if self.acquisition_thread:
            if isinstance(self.acquisition_thread, ArduinoAcquisitionThread):
                self.acquisition_thread.stop()  # 使用专门的停止方法
            self.acquisition_thread.running = False
            self.acquisition_thread.quit()
            self.acquisition_thread.wait()
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        self.update_timer.stop()
        self.status_bar.showMessage("Acquisition stopped")

    def _show_save_dialog(self):
        dialog = SaveConfigDialog(self.active_channels)
        if dialog.exec_() == QDialog.Accepted:
            self.save_channels = dialog.selected_channels()
            self._save_data()

    def _save_data(self):
        if not self.save_channels:
            return
        
        # 在主线程中获取保存目录
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
        
        # 创建并启动保存线程
        self.save_thread = SaveThread(self.data, self.scope, self.save_channels, save_dir)
        self.save_thread.finished.connect(self._on_save_completed)
        self.save_thread.start()
        
        # 禁用保存按钮，防止重复操作
        self.btn_save.setEnabled(False)
        self.status_bar.showMessage("Saving selected data in background...")
    
    def _on_save_completed(self, success):
        # 保存完成后的处理
        self.btn_save.setEnabled(True)
        if success:
            self.status_bar.showMessage("Data saved successfully!")
        else:
            self.status_bar.showMessage("Error occurred while saving data!")

    def _force_redraw(self):
        """ 强制刷新图表显示 """
        for ch in self.active_channels:
            self.subplots[ch].enableAutoRange()
            self.subplots[ch].replot()

    def _update_nanowire_channels(self):
        """Update nanowire controller channel settings"""
        self.nanowire_controller.set_channels(
            self.movement_ch_spin.value(),
            self.direction_ch_spin.value(),
            self.axis_ch_spin.value()
        )
    
    def _update_voltage_threshold(self, channel_type):
        """Update nanowire controller voltage threshold"""
        if channel_type == 'movement':
            self.nanowire_controller.set_voltage_threshold(self.movement_threshold_spin.value(), 'movement')
            print(f"movement threshold {self.movement_threshold_spin.value()}")
        elif channel_type == 'direction':
            self.nanowire_controller.set_voltage_threshold(self.direction_threshold_spin.value(), 'direction')
        elif channel_type == 'axis':
            self.nanowire_controller.set_voltage_threshold(self.axis_threshold_spin.value(), 'axis')
    
    def _update_movement_voltage(self):
        """Update nanowire controller movement voltage"""
        self.nanowire_controller.set_movement_voltage(self.movement_voltage_spin.value())
    
    def _update_nanowire_test_mode(self, state):
        """Update nanowire controller test mode based on checkbox state"""
        is_test_mode = state == Qt.Checked
        self.nanowire_controller.set_test_mode(is_test_mode)
        
        # Update status labels
        status = self.nanowire_controller.get_status()
        self._update_nanowire_status(status)
        
    def _update_nanowire_status(self, status):
        """Update nanowire control status display"""
        # Update movement status
        self.moving_label.setText("Moving" if status['is_moving'] else "Not Moving")
        
        # Update direction status
        self.direction_label.setText(status['direction'])
        
        # Update axis status
        self.axis_label.setText(status['axis'])
        
        # Update voltage status
        self.voltage_label.setText(f"{status['voltage']:.3f} V")
    
    def _force_update_channels(self):
        """强制更新纳米线控制器的通道设置"""
        self._update_nanowire_channels()
        self._update_voltage_threshold('movement')
        self._update_voltage_threshold('direction')
        self._update_voltage_threshold('axis')
        self._update_movement_voltage()
        self._update_movement_step_size() 
        print(f"Force update channels")
        # 获取并更新状态显示
        status = self.nanowire_controller.get_status()
        self._update_nanowire_status(status)
        self.status_bar.showMessage("Nanowire channels updated")



    ## PID control slots 
    def _update_pid_mode(self, state):
        """Update PID control mode based on checkbox state"""
        is_pid_mode = state == Qt.Checked
        
        # 更新控制器PID模式
        self.nanowire_controller.enable_pid_mode(is_pid_mode)
        
        # 启用/禁用目标位置控件
        self.target_x_spin.setEnabled(is_pid_mode)
        self.target_y_spin.setEnabled(is_pid_mode)
        self.btn_set_target.setEnabled(is_pid_mode)
        
        # 启用/禁用线性控制状态显示
        status_display = self.findChild(QGroupBox, "Linear Control Status")
        if status_display:
            status_display.setEnabled(not is_pid_mode)
        
        # 如果PID模式启用，启动位置更新定时器
        if is_pid_mode:
            # 初始化目标位置
            if self.nanowire_test_mode_cb.isChecked():
                # 测试模式下使用默认位置
                current_pos = {'x': 10, 'y': 10, 'theta': 0.0}
            else:
                # 非测试模式从控制器获取位置
                current_pos = self.nanowire_controller.get_current_position()
            
            self.target_x_spin.setValue(current_pos['x'])
            self.target_y_spin.setValue(current_pos['y'])
            
            # 初始化目标轨迹
            if not hasattr(self, 'target_trajectory'):
                self.target_trajectory = []
            else:
                self.target_trajectory = []  # 清空现有轨迹
                
            # 添加初始目标位置到轨迹
            self.target_trajectory.append((time.time(), current_pos['x'], current_pos['y']))
            
            # 如果定时器未运行，启动定时器
            if not hasattr(self, 'position_timer') or not self.position_timer.isActive():
                self.position_timer = QTimer()
                self.position_timer.timeout.connect(self._update_position_display)
                self.position_timer.start(100)  # 每100ms更新一次
        else:
            # 停止位置更新定时器
            if hasattr(self, 'position_timer') and self.position_timer.isActive():
                self.position_timer.stop() 

    def _update_movement_step_size(self):
        """更新移动步长"""
        step_size = self.step_size_spin.value()
        self.nanowire_controller._update_step_pixel_setting(step_size)
    
    def _update_position_display(self):
        """更新位置显示和轨迹可视化"""
        # 如果PID模式未启用，直接返回
        if not self.pid_mode_cb.isChecked():
            return
        
        # 获取当前位置
        if self.nanowire_test_mode_cb.isChecked():
            # 测试模式下，使用模拟位置
            # 如果有目标位置，则模拟向目标位置移动
            if hasattr(self, 'target_trajectory') and self.target_trajectory:
                # 获取最新的目标位置
                latest_target = self.target_trajectory[-1]
                target_x = latest_target[1]
                target_y = latest_target[2]
                
                # 获取当前位置（如果控制器中没有，则使用默认值）
                if not hasattr(self.nanowire_controller, 'current_position'):
                    self.nanowire_controller.current_position = {'x': 10, 'y': 10, 'theta': 0.0}
                
                current_pos = self.nanowire_controller.current_position
                
                # 简单模拟向目标位置移动（每次更新移动一小步）
                step_size = 10  # 每次移动的像素数
                
                # 计算方向向量
                dx = target_x - current_pos['x']
                dy = target_y - current_pos['y']
                
                # 计算距离
                distance = (dx**2 + dy**2)**0.5
                
                if distance > step_size:
                    # 标准化方向向量并乘以步长
                    move_x = dx / distance * step_size
                    move_y = dy / distance * step_size
                    
                    # 更新当前位置
                    new_x = current_pos['x'] + move_x
                    new_y = current_pos['y'] + move_y
                    
                    # 计算角度（弧度转度）
                    theta = np.arctan2(dy, dx) * 180 / np.pi
                    
                    # 更新控制器中的位置
                    self.nanowire_controller.current_position = {'x': new_x, 'y': new_y, 'theta': theta}
                    
                    # 添加到位置历史
                    if not hasattr(self.nanowire_controller, 'position_history'):
                        self.nanowire_controller.position_history = []
                    
                    self.nanowire_controller.position_history.append((time.time(), new_x, new_y))
                    
                    # 限制历史记录长度
                    max_history = 1000
                    if len(self.nanowire_controller.position_history) > max_history:
                        self.nanowire_controller.position_history = self.nanowire_controller.position_history[-max_history:]
                    
                    current_pos = self.nanowire_controller.current_position
                else:
                    # 已经非常接近目标，直接设置为目标位置
                    self.nanowire_controller.current_position = {'x': target_x, 'y': target_y, 'theta': current_pos['theta']}
                    
                    # 添加到位置历史
                    if not hasattr(self.nanowire_controller, 'position_history'):
                        self.nanowire_controller.position_history = []
                    
                    self.nanowire_controller.position_history.append((time.time(), target_x, target_y))
                    
                    current_pos = self.nanowire_controller.current_position
            else:
                # 没有目标位置，使用当前位置
                if not hasattr(self.nanowire_controller, 'current_position'):
                    self.nanowire_controller.current_position = {'x': 10, 'y': 10, 'theta': 0.0}
                current_pos = self.nanowire_controller.current_position
        else:
            # 非测试模式，从控制器获取实际位置
            current_pos = self.nanowire_controller.get_current_position()
        
        # 更新位置标签
        self.current_x_label.setText(str(int(current_pos['x'])))
        self.current_y_label.setText(str(int(current_pos['y'])))
        self.current_theta_label.setText(f"{current_pos['theta']:.1f}°")
        
        # 更新轨迹可视化
        self._update_trajectory_plot()
    
    def _update_target_position(self):
        """当微调框值变化时更新目标位置"""
        if not self.pid_mode_cb.isChecked():
            return
        
        x = self.target_x_spin.value()
        y = self.target_y_spin.value()
        
        # 如果不是测试模式，更新控制器目标位置
        if not self.nanowire_test_mode_cb.isChecked():
            self.nanowire_controller.set_target_position(x, y)
        else:
            # 测试模式下，直接在控制器中设置目标位置
            if not hasattr(self.nanowire_controller, 'target_position'):
                self.nanowire_controller.target_position = {'x': x, 'y': y}
            else:
                self.nanowire_controller.target_position['x'] = x
                self.nanowire_controller.target_position['y'] = y
    
    def _set_target_position(self):
        """点击按钮时设置目标位置"""
        x = self.target_x_spin.value()
        y = self.target_y_spin.value()
        
        # 如果不是测试模式，更新控制器目标位置
        if not self.nanowire_test_mode_cb.isChecked():
            self.nanowire_controller.set_target_position(x, y)
        else:
            # 测试模式下，直接在控制器中设置目标位置
            if not hasattr(self.nanowire_controller, 'target_position'):
                self.nanowire_controller.target_position = {'x': x, 'y': y}
            else:
                self.nanowire_controller.target_position['x'] = x
                self.nanowire_controller.target_position['y'] = y
        
        # 添加目标位置到轨迹
        if not hasattr(self, 'target_trajectory'):
            self.target_trajectory = []
        
        # 获取目标位置
        if self.nanowire_test_mode_cb.isChecked():
            target_pos = {'x': x, 'y': y}
        else:
            target_pos = self.nanowire_controller.get_target_position()
        
        self.target_trajectory.append((time.time(), target_pos['x'], target_pos['y']))
        
        # 更新轨迹图
        self._update_trajectory_plot()
    
    def _update_trajectory_plot(self):
        """更新轨迹可视化"""
        # 获取位置历史
        if self.nanowire_test_mode_cb.isChecked():
            # 测试模式下，从控制器的position_history属性获取
            if hasattr(self.nanowire_controller, 'position_history') and self.nanowire_controller.position_history:
                position_history = self.nanowire_controller.position_history
            else:
                position_history = []
        else:
            # 非测试模式，从控制器方法获取
            position_history = self.nanowire_controller.get_position_history()
        
        if position_history:
            # 提取实际路径的x和y坐标
            times_actual = [p[0] for p in position_history]
            x_actual = [p[1] for p in position_history]
            y_actual = [p[2] for p in position_history]
            
            # 更新实际路径曲线
            self.actual_path.setData(x_actual, y_actual)
            
            # 更新当前位置标记
            self.current_pos_marker.setData([x_actual[-1]], [y_actual[-1]])
        
        # 更新目标路径（如果有）
        if hasattr(self, 'target_trajectory') and self.target_trajectory:
            # 提取目标路径的x和y坐标
            times_target = [p[0] for p in self.target_trajectory]
            x_target = [p[1] for p in self.target_trajectory]
            y_target = [p[2] for p in self.target_trajectory]
            
            # 更新目标路径曲线
            self.target_path.setData(x_target, y_target)
    
    def _clear_trajectory(self):
        """清除轨迹可视化"""
        # 清除轨迹数据
        if hasattr(self, 'target_trajectory'):
            self.target_trajectory = []
        
        # 清除控制器位置历史
        if self.nanowire_test_mode_cb.isChecked():
            # 测试模式下，直接清除控制器的position_history属性
            if hasattr(self.nanowire_controller, 'position_history'):
                self.nanowire_controller.position_history = []
        else:
            # 非测试模式，使用控制器方法清除
            self.nanowire_controller.position_history = []
        
        # 清除轨迹图
        self.target_path.setData([], [])
        self.actual_path.setData([], [])
        self.current_pos_marker.setData([], [])

    ## voltage speed mapping
    def _start_baseline_collection(self, state):
        """开始基线采集"""
        if state == 'low':
            self.btn_collect_low.setEnabled(False)
            self.btn_collect_high.setEnabled(True)
        else:
            self.btn_collect_high.setEnabled(False)
        
        self.nanowire_controller.start_baseline_collection(state)
        QTimer.singleShot(10000, lambda: self._stop_baseline_collection(state))
        self.mapping_status.setText(f"Collecting {state} baseline...")

    def _stop_baseline_collection(self, state):
        """停止基线采集"""
        self.nanowire_controller.stop_baseline_collection(state)
        self.btn_collect_low.setEnabled(True)
        self.btn_collect_high.setEnabled(True)
        
        if state == 'high':
            self.mapping_status.setText("Mapping Configured")
        else:
            self.mapping_status.setText("Waiting for high baseline...")

    def _update_mapping_enable(self, state):
        """启用或禁用电压映射功能"""
        is_enabled = state == Qt.Checked
        
        # 更新控件状态
        self.mapping_ch_spin.setEnabled(is_enabled)
        self.min_step_spin.setEnabled(is_enabled)
        self.max_step_spin.setEnabled(is_enabled)
        self.btn_collect_low.setEnabled(is_enabled)
        self.btn_collect_high.setEnabled(is_enabled)
        
        # 禁用或启用原始步长设置
        self.step_size_spin.setEnabled(not is_enabled)
        
        # 更新控制器映射状态
        if hasattr(self.nanowire_controller, 'enable_voltage_mapping'):
            self.nanowire_controller.enable_voltage_mapping(is_enabled)
            
        if is_enabled:
            self.mapping_status.setText("Mapping: Enabled (Not Configured)")
        else:
            self.mapping_status.setText("Mapping: Disabled")
            
    def _update_mapping_range(self):
        """更新映射范围"""
        min_step = self.min_step_spin.value()
        max_step = self.max_step_spin.value()
        
        # 确保最小值不大于最大值
        if min_step > max_step:
            self.min_step_spin.setValue(max_step)
            min_step = max_step
            
        # 更新控制器映射范围
        if hasattr(self.nanowire_controller, 'set_mapping_range'):
            self.nanowire_controller.set_mapping_range(min_step, max_step)

    def _update_test_mode(self, state):
        # Enable/disable test mode controls
        is_test_mode = state == Qt.Checked
        self.wave_type.setEnabled(is_test_mode)
        self.min_amplitude_spin.setEnabled(is_test_mode)
        self.max_amplitude_spin.setEnabled(is_test_mode)
        self.frequency_spin.setEnabled(is_test_mode)
        self.btn_test_trigger.setEnabled(is_test_mode)

    def _trigger_test_signal(self):
        # Generate test voltages for nanowire control
        test_voltages = {}
        for ch in range(1, 5):
            if ch == self.movement_ch_spin.value():
                test_voltages[ch] = self.movement_voltage_spin.value()
            elif ch == self.direction_ch_spin.value():
                test_voltages[ch] = self.threshold_spin.value() + 0.1  # Ensure above threshold
            elif ch == self.axis_ch_spin.value():
                test_voltages[ch] = self.threshold_spin.value() + 0.1  # Ensure above threshold
            else:
                test_voltages[ch] = 0.0

        # Update nanowire status with test voltages
        self._update_nanowire_status(test_voltages)

    def _start_data_collection(self):
        """开始数据收集"""
        if not self.active_channels:
            self.status_bar.showMessage("Please select at least one channel!")
            return
        
        self.collecting = True
        self.collection_data = {ch: [] for ch in self.active_channels}
        self.btn_collect_start.setEnabled(False)
        self.btn_collect_stop.setEnabled(True)
        self.status_bar.showMessage("Data collection started...")

        # 如果是Arduino模式，发送开始命令
        if self.data_source == "arduino" and self.arduino:
            if not self.arduino.start_acquisition():
                self.status_bar.showMessage("Failed to start Arduino recording")
                return
        
        self.status_bar.showMessage("Data collection started...")
    
    def _stop_data_collection(self):
        """停止数据收集并保存数据"""
        if not self.collecting:
            return
            
        self.collecting = False

        # # 如果是Arduino模式，发送停止命令
        # if self.data_source == "arduino" and self.arduino:
        #     if not self.arduino.stop_acquisition():
        #         self.status_bar.showMessage("Failed to stop Arduino recording")
        #         return

        self.btn_collect_start.setEnabled(True)
        self.btn_collect_stop.setEnabled(False)
        
        # 确保数据目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 弹出标注对话框
        label, ok = QInputDialog.getText(self, 'Data Label', 'Enter label for the collected data:')
        if ok and label:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 在data_dir下创建标注子目录
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            
            # 保存每个通道的数据到标注子目录
            saved_count = 0
            for ch in self.active_channels:
                if ch in self.collection_data and self.collection_data[ch]:
                    filename = os.path.join(label_dir, f'CH{ch}_{timestamp}.csv')
                    try:
                        data_array = np.array(self.collection_data[ch])
                        np.savetxt(filename, data_array,
                                  delimiter=',',
                                  header='Time(s),Voltage(V)',
                                  comments='',
                                  fmt=['%.6f', '%.6f'])
                        saved_count += 1
                    except Exception as e:
                        print(f"Error saving channel {ch} data: {e}")
            
            if saved_count > 0:
                self.status_bar.showMessage(f"Saved data from {saved_count} channels to {label_dir}")
            else:
                self.status_bar.showMessage("No data was saved, collection data might be empty")
        else:
            self.status_bar.showMessage("Data collection cancelled")
        
        # 清空采集数据
        self.collection_data = {}
    
    def _change_storage_path(self):
        new_path = QFileDialog.getExistingDirectory(self, "Select Storage Directory", self.data_dir)
        if new_path:
            self.data_dir = new_path
            self.path_label.setText(f"Current path: {self.data_dir}")
            self.status_bar.showMessage(f"Storage path changed to: {new_path}")

    def _update_test_mode(self, state):
        """更新测试模式状态"""
        is_enabled = state == Qt.Checked
        self.wave_type.setEnabled(is_enabled)
        self.min_amplitude_spin.setEnabled(is_enabled)
        self.max_amplitude_spin.setEnabled(is_enabled)
        self.frequency_spin.setEnabled(is_enabled)
        self.btn_test_trigger.setEnabled(is_enabled)
        
        # 如果关闭测试模式，确保停止当前的测试信号
        if not is_enabled and self.running and isinstance(self.acquisition_thread, TestAcquisitionThread):
            self._stop_acquisition()

    def _trigger_test_signal(self):
        """触发测试信号"""
        if not self.test_mode_cb.isChecked() or not self.active_channels:
            return
            
        try:
            # 如果已经在运行，先停止当前采集
            if self.running:
                self._stop_acquisition()
            
            # 启动新的测试采集
            self._start_acquisition()
            self.status_bar.showMessage("Test signal acquisition started")
        except Exception as e:
            print(f"Error triggering test signal: {e}")
            self.status_bar.showMessage(f"Error: {str(e)}")

    def closeEvent(self, event):
        self._stop_acquisition()
        if self.scope:
            self.scope.close()
        event.accept()

# -------------------- 程序入口 --------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置全局字体
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = OscilloscopeGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()