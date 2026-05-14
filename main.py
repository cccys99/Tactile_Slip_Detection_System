import sys
import numpy as np
import pyqtgraph as pg 
import serial.tools.list_ports
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.QtCore import Qt

from UI.Ui_main_window import Ui_MainWindow
from hardware_io import SensorDataThread

class TactileSlipSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("异构多模态触觉滑移检测与闭环控制系统")
        
        # 在初始化绘图时，预先配置 pyqtgraph 的高性能 FIFO 实现
        self.init_plots_hardware_acceleration()
        
        self.sensor_thread = SensorDataThread()
        # 替换新的解耦汇报信号
        self.sensor_thread.dashboard_update_signal.connect(self.update_dashboard_optimized)
        
        self.bind_signals()
        self.scan_ports()

    def scan_ports(self):
        self.ui.comboBox.clear() 
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.ui.comboBox.addItem(f"{port.device} - {port.description}", port.device)
        if ports:
            self.ui.text_log.append(f">> [系统初始化] 扫描到 {len(ports)} 个可用串口设备。")
        else:
            self.ui.text_log.append(">> [系统警告] 未扫描到任何串口设备，请检查 USB 连接！")

    def init_plots_hardware_acceleration(self):
        """配置 pyqtgraph 的高速、流式、FIFO 绘图"""
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        # pyqtgraph 优化：开启图表抗锯齿和追加绘图模式
        pg.setConfigOption('antialias', True) # 牺牲一点算力，换取丝滑波形
        
        # ==================== 1. 配置气压高性能 FIFO 绘图 ====================
        self.layout_p = QVBoxLayout(self.ui.plot_pressure)
        self.layout_p.setContentsMargins(0, 0, 0, 0)
        # pyqtgraph 的核心 FIFO 控件 (不需要预填充矩阵)
        self.pw_pressure = pg.PlotWidget(title="气压法向力特征 (4阵列)")
        self.pw_pressure.addLegend(offset=(10, 10)) 
        # self.pw_pressure.enableAutoRange(axis='y') # Y轴根据一整块数据自动缩放
        self.pw_pressure.setYRange(95000, 135000)
        # 使能高性能追加模式，彻底避免重绘卡死
        self.pw_pressure.setDownsampling(mode='peak') # 对历史数据降采样
        self.pw_pressure.setClipToView(True) # 只渲染视图内的点
        self.layout_p.addWidget(self.pw_pressure)
        
        p_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
        p_names = ['P1', 'P2', 'P3', 'P4']
        self.curves_p = []
        for i in range(4):
            # 创建绘图对象，但不预填充数据
            curve = self.pw_pressure.plot(pen=pg.mkPen(p_colors[i], width=2), name=p_names[i])
            self.curves_p.append(curve)
        
        # ==================== 2. 配置IMU高性能 FIFO 绘图 ====================
        self.layout_i = QVBoxLayout(self.ui.plot_imu)
        self.layout_i.setContentsMargins(0, 0, 0, 0)
        self.pw_imu = pg.PlotWidget(title="IMU瞬态高频特征 (6轴)")
        self.pw_imu.addLegend(offset=(10, 10)) 
        # self.pw_imu.enableAutoRange(axis='y')
        self.pw_imu.setYRange(-800, 800)
        self.pw_imu.setDownsampling(mode='peak')
        self.pw_imu.setClipToView(True)
        self.layout_i.addWidget(self.pw_imu)
        
        i_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
        i_names = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
        self.curves_i = []
        for i in range(6):
            if i < 3: 
                pen = pg.mkPen(i_colors[i], width=1.5)
            else:     
                pen = pg.mkPen(i_colors[i], width=1.5, style=Qt.DashLine)
            curve = self.pw_imu.plot(pen=pen, name=i_names[i])
            self.curves_i.append(curve)

    def bind_signals(self):
        self.ui.btn_connect_serial.clicked.connect(self.start_sensor_stream)

    def start_sensor_stream(self):
        if not self.sensor_thread.isRunning():
            port_name = self.ui.comboBox.currentData() 
            if not port_name:
                self.ui.text_log.append(">> [错误] 未检测到串口，请检查连接！")
                return
            
            self.sensor_thread.port_name = port_name 
            self.sensor_thread.start()
            
            self.ui.btn_connect_serial.setText("关闭串口")
            self.ui.text_log.append(f">> 正在连接 {port_name} 读取 1M 实时数据...")
        else:
            self.sensor_thread.stop()
            self.ui.btn_connect_serial.setText("打开串口")
            self.ui.text_log.append(">> 已断开连接。")

    def update_dashboard_optimized(self, dashboard_update):
        # 1. 提取子线程准备好的一整块 X 和 10 通道 Y 矩阵
        # time_arr (N 帧)
        # p_matrix (4 通道 x N 帧)
        # i_matrix (6 通道 x N 帧)
        time_x = dashboard_update["time_arr"]
        
        # 2. 气压波形流式追加
        # pyqtgraph 的 curve.setData 在收到 x, y 数组并且开启 clipping 时极其高效
        for j in range(4):
            self.curves_p[j].setData(time_x, dashboard_update["p_matrix"][j])
        
        # 3. IMU 波形流式追加
        for j in range(6):
            self.curves_i[j].setData(time_x, dashboard_update["i_matrix"][j])
        
        # 更新中间文字状态
        prob_val = dashboard_update["prob_all"]
        self.ui.bar_prob_press.setValue(prob_val)
        self.ui.bar_prob_imu.setValue(prob_val)
        self.ui.bar_prob_final.setValue(prob_val)
        self.ui.label_k_value.setText(f"冲突因子 K 值: {dashboard_update['k_value']}")
        
        current_state = dashboard_update['state_str']
        self.ui.label_state_press.setText(f"状态: {current_state}")
        self.ui.label_state_imu.setText(f"状态: {current_state}")
        self.ui.label_state_final.setText(f"最终判定: {current_state}")

        # 右侧指示灯魔法
        dark_style = "background-color: #E0E0E0; color: #808080; border-radius: 5px; padding: 5px;"
        active_style = "color: white; font-weight: bold; border-radius: 5px; padding: 5px;"
        
        self.ui.light_static.setStyleSheet(dark_style)
        self.ui.light_move.setStyleSheet(dark_style)
        self.ui.light_friction.setStyleSheet(dark_style)
        self.ui.light_impact.setStyleSheet(dark_style)

        if current_state == "静止按压":
            self.ui.light_static.setStyleSheet(f"background-color: #34C759; {active_style}")
        elif current_state == "空载移动":
            self.ui.light_move.setStyleSheet(f"background-color: #007AFF; {active_style}")
        elif current_state == "摩擦滑移":
            self.ui.light_friction.setStyleSheet(f"background-color: #FF3B30; {active_style}")
        elif current_state == "碰撞冲击":
            self.ui.light_impact.setStyleSheet(f"background-color: #FF9500; {active_style}")

    def closeEvent(self, event):
        self.sensor_thread.stop()
        event.accept()

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = TactileSlipSystem()
    window.show()
    sys.exit(app.exec_())