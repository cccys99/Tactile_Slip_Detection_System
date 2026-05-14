import time
import serial
import numpy as np  
import pywt
import joblib
import scipy.signal as signal
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor 

# ================= 1. 离线复刻滤波 =================
def apply_realtime_filters(p_matrix, i_matrix):
    """
    对传入的长窗口(256帧)进行整体滤波，保证 1Hz 滤波器有足够的数据平滑去重力。
    """
    FS = 200
    
    # 1. 气压低通 (20Hz)
    b_lp, a_lp = signal.butter(4, 20 / (FS / 2), 'low')
    p_filt = np.zeros_like(p_matrix)
    for col in range(4):
        p_filt[:, col] = signal.filtfilt(b_lp, a_lp, p_matrix[:, col], padlen=15)
        
    # 2. 纯正的 IMU 高通 (1Hz) -> 复刻离线逻辑
    b_hp, a_hp = signal.butter(4, 1 / (FS / 2), 'high')
    i_filt = np.zeros_like(i_matrix)
    for col in range(6):
        i_filt[:, col] = signal.filtfilt(b_hp, a_hp, i_matrix[:, col], padlen=15)
        
    return p_filt, i_filt

# ================= 2. 小波与特征提取 =================
def get_dwt_features(data_array):
    coeffs = pywt.wavedec(data_array, 'db4', level=3)
    features = []
    for coeff in coeffs:
        energy = np.log(np.sum(np.square(coeff)) + 1e-10)
        features.append(energy)
        p = np.square(coeff) / (np.sum(np.square(coeff)) + 1e-10)
        entropy = -np.sum(p * np.log2(p + 1e-10))
        features.append(entropy)
    return features

def extract_features_dual(window_list):
    # 将字典列表转为矩阵 (此时长度是 256 帧)
    p_matrix = np.array([[d['P1_Pa'], d['P2_Pa'], d['P3_Pa'], d['P4_Pa']] for d in window_list])
    i_matrix = np.array([[d['AccX_g'], d['AccY_g'], d['AccZ_g'], d['GyroX_dps'], d['GyroY_dps'], d['GyroZ_dps']] for d in window_list])
    
    # 1. 对 256 帧进行整体滤波
    p_filt, i_filt = apply_realtime_filters(p_matrix, i_matrix)
    
    # 2. 关键：只截取最后的 64 帧作为当前特征窗口
    p_window = p_filt[-64:, :]
    imu_window = i_filt[-64:, :]
    
    # --- 提取气压特征 ---
    p_feats = []
    p_feats.extend(np.mean(p_window, axis=0))
    p_feats.extend(np.std(p_window, axis=0))
    for col in range(p_window.shape[1]):
        p_feats.extend(get_dwt_features(p_window[:, col]))
        
    # --- 提取IMU特征 ---
    imu_feats = []
    for col in range(imu_window.shape[1]):
        imu_feats.extend(get_dwt_features(imu_window[:, col]))
        
    return np.array(p_feats).reshape(1, -1), np.array(imu_feats).reshape(1, -1)

# ================= 3. D-S 证据理论 =================
def ds_combine_realtime(prob_a, prob_b):
    n_classes = len(prob_a)
    mass_fused = np.zeros(n_classes)
    conflict_k = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            product = prob_a[i] * prob_b[j]
            if i == j: mass_fused[i] += product
            else: conflict_k += product
    if conflict_k < 0.9999: mass_fused = mass_fused / (1 - conflict_k)
    else: mass_fused = (prob_a + prob_b) / 2
    return mass_fused, conflict_k

class SensorDataThread(QThread):
    dashboard_update_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.port_name = None       
        self.baudrate = 1000000      
        
        try:
            self.rf_pressure = joblib.load("rf_pressure.pkl")
            self.rf_imu = joblib.load("rf_imu.pkl")
            print("[子线程] ✅ 成功加载双模态模型")
            self.models_loaded = True
        except Exception as e:
            self.models_loaded = False
            
        self.state_map = {0: "静止按压", 1: "空载移动", 2: "发生摩擦", 3: "碰撞冲击"}
        self.ai_executor = ThreadPoolExecutor(max_workers=1)
        self.last_final_state = "数据缓冲中..." 
        self.last_prob_val = 0
        self.last_k_value = 0.0  
        self.vote_buffer = deque(maxlen=3) # 稍微增加一点平滑度
        self.is_ai_busy = False

        # 气压硬件公差校准变量（必须保留，否则不同传感器气压绝对值对不上）
        self.is_calibrated = False
        self.calibration_frames = 0
        self.base_p = np.zeros(4) 
        self.TRAIN_BASELINE = 102418.0 

    def _run_ai_task(self, window_data_copy):
        try:
            if not self.models_loaded: return
            
            feats_p, feats_i = extract_features_dual(window_data_copy)
            probs_p = self.rf_pressure.predict_proba(feats_p)[0]
            probs_i = self.rf_imu.predict_proba(feats_i)[0]
            
            fused_prob, k_val = ds_combine_realtime(probs_p, probs_i)
            self.last_prob_val = int(np.max(fused_prob) * 100)
            self.last_k_value = round(k_val, 3) 
            
            pred_class = np.argmax(fused_prob)
            self.vote_buffer.append(pred_class)
            
            final_class = max(set(self.vote_buffer), key=self.vote_buffer.count)
            self.last_final_state = self.state_map[final_class]
            
        except Exception as e:
            pass
        finally:
            self.is_ai_busy = False

    def run(self):
        self.is_running = True 
        try:
            ser = serial.Serial(self.port_name, self.baudrate, timeout=1)
            ser.set_buffer_size(rx_size=1024*1024, tx_size=1024) 
        except Exception as e:
            return

        self.internal_t = deque(maxlen=200)
        self.internal_p = deque(maxlen=200) 
        self.internal_i = deque(maxlen=200) 
        
        # 扩大缓冲池为 256 帧（1.28秒）
        window_buffer = deque(maxlen=256) 
        
        start_time_ms = None 
        last_report_time = time.time()
        self.ui_update_interval = 0.033 
        frame_counter = 0
        leftover_bytes = b''

        while self.is_running:
            try:
                waiting = ser.in_waiting
                if waiting > 0:
                    chunk = ser.read(waiting)
                    leftover_bytes += chunk
                    lines = leftover_bytes.split(b'\n')
                    leftover_bytes = lines.pop() 
                    
                    for raw_line in lines:
                        raw_line = raw_line.strip() 
                        if not raw_line: continue
                            
                        parts = raw_line.split(b',')
                        if len(parts) == 11: 
                            try:
                                raw_time_ms = float(parts[0])
                                if start_time_ms is None: start_time_ms = raw_time_ms 
                                current_time_s = (raw_time_ms - start_time_ms) / 1000.0

                                raw_p = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                                raw_i = np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9]), float(parts[10])])
                                
                                # 开机 0.5 秒自动对齐气压基线
                                if not self.is_calibrated:
                                    self.base_p += raw_p
                                    self.calibration_frames += 1
                                    if self.calibration_frames >= 100:
                                        self.base_p /= 100.0 
                                        self.is_calibrated = True
                                        print(f"\n[系统校准] 气压基准已平移对齐至: {self.TRAIN_BASELINE}")
                                    adj_p = np.array([self.TRAIN_BASELINE] * 4)
                                else:
                                    adj_p = (raw_p - self.base_p) + self.TRAIN_BASELINE

                                row_dict = {
                                    'P1_Pa': adj_p[0], 'P2_Pa': adj_p[1], 'P3_Pa': adj_p[2], 'P4_Pa': adj_p[3],
                                    'AccX_g': raw_i[0], 'AccY_g': raw_i[1], 'AccZ_g': raw_i[2], 
                                    'GyroX_dps': raw_i[3], 'GyroY_dps': raw_i[4], 'GyroZ_dps': raw_i[5]
                                }
                                
                                i_frame = [raw_i[0]*100, raw_i[1]*100, raw_i[2]*100, raw_i[3], raw_i[4], raw_i[5]]
                                
                                self.internal_t.append(current_time_s)
                                self.internal_p.append(adj_p.tolist())
                                self.internal_i.append(i_frame)
                                window_buffer.append(row_dict)
                                
                                frame_counter += 1
                            except ValueError:
                                continue
                else:
                    time.sleep(0.001)

                current_loop_time = time.time()
                if current_loop_time - last_report_time > self.ui_update_interval:
                    if len(self.internal_t) > 2:
                        dashboard_update = {
                            "time_arr": np.array(self.internal_t),
                            "p_matrix": np.array(self.internal_p).T,
                            "i_matrix": np.array(self.internal_i).T,
                            "k_value": self.last_k_value,
                            "prob_all": self.last_prob_val,        
                            "state_str": self.last_final_state     
                        }
                        self.dashboard_update_signal.emit(dashboard_update)
                    last_report_time = current_loop_time

                # 当缓冲池满 256 帧时才开始 AI 推理，保证滤波器有充分数据
                if len(window_buffer) == 256 and self.models_loaded:  
                    if frame_counter % 5 == 0 and not self.is_ai_busy: 
                        self.is_ai_busy = True 
                        self.ai_executor.submit(self._run_ai_task, list(window_buffer))

            except Exception as e:
                continue

        if ser and ser.is_open:
            ser.close()
        self.ai_executor.shutdown(wait=False)

    def stop(self):
        self.is_running = False
        self.wait()