import time
import pandas as pd
import numpy as np
import pywt
import joblib
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal

def extract_features(window_df):
    feats = []
    # --- A. 气压特征 ---
    for col in ['P1_Pa', 'P2_Pa', 'P3_Pa', 'P4_Pa']:
        sig = window_df[col].values
        feats.append(np.mean(sig))
        feats.append(np.std(sig))
        feats.append(np.ptp(sig))
        try:
            coeffs = pywt.wavedec(sig, 'db4', level=2)
            feats.append(np.sum(np.square(coeffs[1]))) # D1
            feats.append(np.sum(np.square(coeffs[2]))) # D2
        except:
            feats.extend([0, 0])

    # --- B. IMU 特征 ---
    imu_cols = ['AccX_g', 'AccY_g', 'AccZ_g', 'GyroX_dps', 'GyroY_dps', 'GyroZ_dps']
    for col in imu_cols:
        sig = window_df[col].values
        feats.append(np.std(sig))
        feats.append(np.mean(np.abs(sig)))

    return np.array(feats).reshape(1, -1)

class SensorDataThread(QThread):
    data_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.is_running = False
        # 请确保文件名与你实际的文件名一致
        self.csv_file = "demo_final.csv"
        self.model_path = "tactile_model_final.pkl"
        
        try:
            self.clf = joblib.load(self.model_path)
            print(f"[子线程] ✅ 成功加载机器学习模型: {self.model_path}")
        except Exception as e:
            print(f"[致命错误] 模型加载失败: {e}")
            self.clf = None
            
        self.state_map = {
            0: "静止按压",
            1: "空载移动",
            2: "发生摩擦",
            3: "碰撞冲击"
        }

    def run(self):
        self.is_running = True 
        
        try:
            df = pd.read_csv(self.csv_file)
            print(f"[子线程] 开始播放离线数据...")
        except FileNotFoundError:
            print(f"[致命错误] 找不到数据文件！")
            return

        window_buffer = deque(maxlen=64) 
        vote_buffer = deque(maxlen=10)   
        
        frame_counter = 0             
        last_final_state = "数据缓冲中..." 
        last_prob_val = 0             

        while self.is_running:
            for index, row in df.iterrows():
                if not self.is_running: break 

                try:
                    # 【修改点】：直接使用原始 Timestamp，不做任何换算
                    raw_time_ms = float(row.iloc[0])

                    row_dict = {
                        'P1_Pa': float(row.iloc[1]), 'P2_Pa': float(row.iloc[2]),
                        'P3_Pa': float(row.iloc[3]), 'P4_Pa': float(row.iloc[4]),
                        'AccX_g': float(row.iloc[5]), 'AccY_g': float(row.iloc[6]),
                        'AccZ_g': float(row.iloc[7]), 'GyroX_dps': float(row.iloc[8]),
                        'GyroY_dps': float(row.iloc[9]), 'GyroZ_dps': float(row.iloc[10])
                    }
                    pressures = [row_dict[f'P{i}_Pa'] for i in range(1, 5)]
                    imus = [row_dict['AccX_g'], row_dict['AccY_g'], row_dict['AccZ_g'], 
                            row_dict['GyroX_dps'], row_dict['GyroY_dps'], row_dict['GyroZ_dps']]
                except Exception as e:
                    continue

                window_buffer.append(row_dict)
                frame_counter += 1 

                # 降频推理：满64帧后，每5帧推理一次
                if len(window_buffer) == 64 and self.clf is not None:
                    if frame_counter % 5 == 0:
                        window_df = pd.DataFrame(window_buffer)
                        feats = extract_features(window_df)
                        
                        pred_class = self.clf.predict(feats)[0]          
                        pred_probs = self.clf.predict_proba(feats)[0]    
                        
                        vote_buffer.append(pred_class)
                        most_common_class = max(set(vote_buffer), key=vote_buffer.count)
                        
                        last_final_state = self.state_map[most_common_class]
                        last_prob_val = int(np.max(pred_probs) * 100)

                mock_data = {
                    "time": raw_time_ms, # 直接传出原始毫秒
                    "val_p": pressures,
                    "val_i": imus,
                    "prob_press": last_prob_val,
                    "prob_imu": last_prob_val,
                    "prob_final": last_prob_val,
                    "k_value": 0.0,
                    "state_final": last_final_state
                }

                self.data_updated.emit(mock_data)
                time.sleep(0.005) 

    def stop(self):
        self.is_running = False
        self.wait()