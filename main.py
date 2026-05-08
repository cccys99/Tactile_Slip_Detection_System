import sys
import numpy as np
import pyqtgraph as pg 
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
        
        self.init_plots()
        
        self.sensor_thread = SensorDataThread()
        self.sensor_thread.data_updated.connect(self.update_dashboard)
        
        self.bind_signals()

    def init_plots(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.layout_p = QVBoxLayout(self.ui.plot_pressure)
        self.layout_p.setContentsMargins(0, 0, 0, 0)
        self.pw_pressure = pg.PlotWidget(title="气压法向力低频特征 (4通道阵列)")
        self.pw_pressure.addLegend(offset=(10, 10)) 
        self.pw_pressure.enableAutoRange(axis='y')
        self.pw_pressure.setLabel('bottom', 'Timestamp') # 恢复为 Timestamp
        self.layout_p.addWidget(self.pw_pressure)
        
        p_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
        p_names = ['P1', 'P2', 'P3', 'P4']
        self.curves_p = []
        for i in range(4):
            curve = self.pw_pressure.plot(pen=pg.mkPen(p_colors[i], width=2), name=p_names[i])
            self.curves_p.append(curve)
        
        self.layout_i = QVBoxLayout(self.ui.plot_imu)
        self.layout_i.setContentsMargins(0, 0, 0, 0)
        self.pw_imu = pg.PlotWidget(title="IMU瞬态高频振动特征 (6轴)")
        self.pw_imu.addLegend(offset=(10, 10)) 
        self.pw_imu.enableAutoRange(axis='y')
        self.pw_imu.setLabel('bottom', 'Timestamp') # 恢复为 Timestamp
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
        
        self.data_t = np.zeros(200)       
        self.data_p = np.zeros((4, 200))  
        self.data_i = np.zeros((6, 200))  
        self.is_first_data = True

    def bind_signals(self):
        self.ui.btn_connect_serial.clicked.connect(self.start_sensor_stream)

    def start_sensor_stream(self):
        if not self.sensor_thread.isRunning():
            self.sensor_thread.start()
            self.ui.btn_connect_serial.setText("停止读取")
            self.ui.text_log.append(">> [离线模式] 启动虚拟传感器数据流...")
        else:
            self.sensor_thread.stop()
            self.ui.btn_connect_serial.setText("打开串口")
            self.ui.text_log.append(">> [离线模式] 已停止读取数据。")

    def update_dashboard(self, data):
        if self.is_first_data:
            self.data_t.fill(data["time"])
            for j in range(4): self.data_p[j].fill(data["val_p"][j])
            for j in range(6): self.data_i[j].fill(data["val_i"][j])
            self.is_first_data = False

        self.data_t[:-1] = self.data_t[1:]
        self.data_t[-1] = data["time"]

        for j in range(4):
            self.data_p[j][:-1] = self.data_p[j][1:]
            self.data_p[j][-1] = data["val_p"][j]
            self.curves_p[j].setData(x=self.data_t, y=self.data_p[j])
        
        for j in range(6):
            self.data_i[j][:-1] = self.data_i[j][1:]
            self.data_i[j][-1] = data["val_i"][j]
            self.curves_i[j].setData(x=self.data_t, y=self.data_i[j])
        
        self.ui.bar_prob_press.setValue(data["prob_press"])
        self.ui.bar_prob_imu.setValue(data["prob_imu"])
        self.ui.bar_prob_final.setValue(data["prob_final"])
        
        self.ui.label_k_value.setText(f"冲突因子 K 值: {data['k_value']}")
        
        current_state = data['state_final']
        self.ui.label_state_press.setText(f"状态: {current_state}")
        self.ui.label_state_imu.setText(f"状态: {current_state}")
        self.ui.label_state_final.setText(f"最终判定: {current_state}")

        # ================= 4. 右侧指示灯魔法（已修正名称）=================
        # 先把所有灯熄灭 (暗灰色背景)
        dark_style = "background-color: #E0E0E0; color: #808080; border-radius: 5px; padding: 5px;"
        
        # 使用截图中的真实命名：light_static, light_move, light_friction, light_impact
        self.ui.light_static.setStyleSheet(dark_style)
        self.ui.light_move.setStyleSheet(dark_style)
        self.ui.light_friction.setStyleSheet(dark_style)
        self.ui.light_impact.setStyleSheet(dark_style)

        # 根据真实模型推断，点亮对应的灯！
        active_style = "color: white; font-weight: bold; border-radius: 5px; padding: 5px;"
        if current_state == "静止按压":
            self.ui.light_static.setStyleSheet(f"background-color: #34C759; {active_style}")
        elif current_state == "空载移动":
            self.ui.light_move.setStyleSheet(f"background-color: #007AFF; {active_style}")
        elif current_state == "发生摩擦":
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