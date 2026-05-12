import time
import serial
import numpy as np  
import pywt
import joblib
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor 

# ================= 极速版特征提取 =================
def extract_features(window_list):
    feats = []
    data = {k: np.array([d[k] for d in window_list]) for k in window_list[0].keys()}

    for col in ['P1_Pa', 'P2_Pa', 'P3_Pa', 'P4_Pa']:
        sig = data[col]
        feats.append(np.mean(sig))
        feats.append(np.std(sig))
        feats.append(np.ptp(sig))
        try:
            coeffs = pywt.wavedec(sig, 'db4', level=2)
            feats.append(np.sum(np.square(coeffs[1]))) 
            feats.append(np.sum(np.square(coeffs[2]))) 
        except:
            feats.extend([0, 0])

    imu_cols = ['AccX_g', 'AccY_g', 'AccZ_g', 'GyroX_dps', 'GyroY_dps', 'GyroZ_dps']
    for col in imu_cols:
        sig = data[col]
        feats.append(np.std(sig))
        feats.append(np.mean(np.abs(sig)))

    return np.array(feats).reshape(1, -1)


class SensorDataThread(QThread):
    dashboard_update_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.port_name = None       
        self.baudrate = 1000000      
        self.model_path = "tactile_model_final.pkl"
        
        try:
            self.clf = joblib.load(self.model_path)
            print(f"[子线程] ✅ 加载模型成功: {self.model_path}")
        except Exception:
            self.clf = None
            
        self.state_map = {0: "静止按压", 1: "空载移动", 2: "发生摩擦", 3: "碰撞冲击"}
        self.ai_executor = ThreadPoolExecutor(max_workers=1)
        
        self.last_final_state = "数据缓冲中..." 
        self.last_prob_val = 0
        self.vote_buffer = deque(maxlen=10)

    def _run_ai_task(self, window_data_copy):
        try:
            feats = extract_features(window_data_copy)
            pred_probs = self.clf.predict_proba(feats)[0]    
            
            self.last_prob_val = int(np.max(pred_probs) * 100)
            pred_class = np.argmax(pred_probs)
            self.vote_buffer.append(pred_class)
            most_common_class = max(set(self.vote_buffer), key=self.vote_buffer.count)
            self.last_final_state = self.state_map[most_common_class]
        except Exception as e:
            pass 

    def run(self):
        self.is_running = True 
        
        try:
            ser = serial.Serial(self.port_name, self.baudrate, timeout=1)
            ser.set_buffer_size(rx_size=1024*1024, tx_size=1024) 
            print(f"[子线程] ✅ 成功连接 1M 串口: {self.port_name}")
        except Exception as e:
            print(f"[致命错误] 串口打开失败: {e}")
            return

        internal_fifo_size = 200 
        self.internal_t = deque(maxlen=internal_fifo_size)
        self.internal_p = deque(maxlen=internal_fifo_size) 
        self.internal_i = deque(maxlen=internal_fifo_size) 
        
        window_buffer = deque(maxlen=64) 
        
        start_time_ms = None 
        last_report_time = time.time()
        self.ui_update_interval = 0.033 
        
        frame_counter = 0

        # =========================================================
        # 🚀 极致性能杀招：二进制块缓存区
        # =========================================================
        leftover_bytes = b''

        while self.is_running:
            try:
                # 1. 瞬间抽空硬件缓冲区，绝不给芯片溢出的机会！
                waiting = ser.in_waiting
                if waiting > 0:
                    # 读取所有积压的字节
                    chunk = ser.read(waiting)
                    leftover_bytes += chunk
                    
                    # 按照二进制换行符分割
                    lines = leftover_bytes.split(b'\n')
                    
                    # 最后一段可能是不完整的半行，保留给下一次拼接
                    leftover_bytes = lines.pop() 
                    
                    # 批量解析这一波收到的所有完整行
                    for raw_line in lines:
                        raw_line = raw_line.strip() # 去掉 \r
                        if not raw_line: 
                            continue
                            
                        # 核心提速：直接以 Byte 格式切割，跳过 utf-8 解码！
                        parts = raw_line.split(b',')
                        
                        if len(parts) == 11: 
                            try:
                                # 核心提速：Python 底层 C 语言直接将字节流转为 float
                                raw_time_ms = float(parts[0])
                                if start_time_ms is None:
                                    start_time_ms = raw_time_ms 
                                current_time_s = (raw_time_ms - start_time_ms) / 1000.0

                                row_dict = {
                                    'P1_Pa': float(parts[1]), 'P2_Pa': float(parts[2]),
                                    'P3_Pa': float(parts[3]), 'P4_Pa': float(parts[4]),
                                    'AccX_g': float(parts[5]), 'AccY_g': float(parts[6]),
                                    'AccZ_g': float(parts[7]), 'GyroX_dps': float(parts[8]),
                                    'GyroY_dps': float(parts[9]), 'GyroZ_dps': float(parts[10])
                                }
                                
                                p_frame = [row_dict[f'P{i}_Pa'] for i in range(1, 5)]
                                i_frame = [row_dict['AccX_g'] * 100, 
                                           row_dict['AccY_g'] * 100, 
                                           row_dict['AccZ_g'] * 100, 
                                           row_dict['GyroX_dps'], 
                                           row_dict['GyroY_dps'], 
                                           row_dict['GyroZ_dps']]
                                
                                self.internal_t.append(current_time_s)
                                self.internal_p.append(p_frame)
                                self.internal_i.append(i_frame)
                                window_buffer.append(row_dict)
                                
                                frame_counter += 1
                                
                            except ValueError:
                                # 如果遇到单片机刚开机时发送的乱码字节，安静地跳过
                                continue
                else:
                    # 如果串口没数据，轻微让出 CPU 时间片，防止单核跑到 100%
                    time.sleep(0.001)

                # =========================================================
                # 2. 定时发送波形给 UI (30Hz) - 不在 for 循环里，防止阻塞读取
                # =========================================================
                current_loop_time = time.time()
                if current_loop_time - last_report_time > self.ui_update_interval:
                    if len(self.internal_t) > 2:
                        dashboard_update = {
                            "time_arr": np.array(self.internal_t),
                            "p_matrix": np.array(self.internal_p).T,
                            "i_matrix": np.array(self.internal_i).T,
                            "k_value": 0.0,
                            "prob_all": self.last_prob_val,        
                            "state_str": self.last_final_state     
                        }
                        self.dashboard_update_signal.emit(dashboard_update)
                    last_report_time = current_loop_time

                # =========================================================
                # 3. 异步 AI 调度
                # =========================================================
                if len(window_buffer) == 64 and self.clf is not None:
                    if frame_counter % 10 == 0: 
                        self.ai_executor.submit(self._run_ai_task, list(window_buffer))

            except Exception as e:
                continue

        if ser and ser.is_open:
            ser.close()
        self.ai_executor.shutdown(wait=False)

    def stop(self):
        self.is_running = False
        self.wait()