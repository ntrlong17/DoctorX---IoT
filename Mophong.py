import cv2
import face_recognition
import os
import numpy as np
from gtts import gTTS
import pygame
import time
import threading 
from ultralytics import YOLO
import speech_recognition as sr
import re 
import pandas as pd 
from datetime import datetime
import requests 
# 1. CẤU HÌNH HỆ THỐNG
PATH_ANH = "C:/IoT/User" 
MODEL_YOLO = YOLO('yolov8n.pt') 
DUNG_TICH_THAT = 250 
DO_NHAY_MIC = 200 
FILE_EXCEL = "C:/IoT/lich_su_uong_nuoc.xlsx" 
BACKEND_URL = "http://127.0.0.1:8000"  # nếu dùng ngrok thì đổi URL này
DEVICE_ID = "water_dev_01"             # đúng device_id trên platform
DEVICE_API_KEY = "b360175f022c830bb918c832b348fc03" 
# 2. KHỞI TẠO BIẾN TOÀN CỤC
pygame.mixer.init()
recog = sr.Recognizer()
db_encs = []
db_names = []

session_data = {
    "name": None,
    "sex": None,
    "height": None,
    "weight": None,
    "total_need": 0
}

def xu_ly_am_thanh(text):
    try:
        filename = f"voice_{int(time.time())}_{np.random.randint(0,100)}.mp3"
        tts = gTTS(text=text, lang='vi')
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(filename)
    except: pass

def noi(text, wait=True):
    print(f"Doctor.X: {text}")
    if wait: xu_ly_am_thanh(text)
    else: threading.Thread(target=xu_ly_am_thanh, args=(text,)).start()

def nghe(mode="so"):
    with sr.Microphone() as source:
        print("\n>> ĐANG LẮNG NGHE...")
        recog.energy_threshold = DO_NHAY_MIC
        recog.pause_threshold = 1.0 
        try:
            audio = recog.listen(source, timeout=8)
            text = recog.recognize_google(audio, language="vi-VN").lower()
            print(f"-> Bạn nói: {text}")
            
            if mode == "so":
                nums = re.findall(r'\d+', text)
                return int(nums[0]) if nums else None
            elif mode == "sex":
                if any(x in text for x in ["nam", "trai", "đàn ông"]): return "Nam"
                if any(x in text for x in ["nữ", "gái", "phụ nữ"]): return "Nữ"
                return None
            elif mode == "dang_ky":
                if any(x in text for x in ["không đăng ký", "không muốn", "hủy"]): return False
                if any(x in text for x in ["tôi đăng ký", "muốn đăng ký", "đồng ý"]): return True
                return None
            elif mode == "ho_ten":
                text = text.replace("tên tôi là", "").replace("tên là", "").strip()
                return text.title() if len(text.split()) >= 2 else None
        except: return None

def nap_du_lieu():
    global db_encs, db_names
    db_encs = []
    db_names = []
    
    print("--- ĐANG NẠP DỮ LIỆU KHUÔN MẶT ---")
    if os.path.exists(PATH_ANH):
        for f in os.listdir(PATH_ANH):
            if f.endswith(('.jpg', '.png')):
                try:
                    path = f"{PATH_ANH}/{f}"
                    img = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(img)
                    
                    if len(encodings) > 0:
                        db_encs.append(encodings[0])
                        name = os.path.splitext(f)[0]
                        db_names.append(name)
                        print(f"-> Đã học: {name}") # In ra để bạn yên tâm
                    else:
                        print(f"-> Bỏ qua {f}: Không thấy mặt.")
                except: pass
    print(f"--- TỔNG CỘNG: {len(db_names)} NGƯỜI ---")
    return db_encs, db_names
# 3. XỬ LÝ EXCEL
COLUMNS = ["Ngay", "Ten", "GioiTinh", "ChieuCao", "CanNang", "CanUong", "Sang", "Trua", "Chieu", "Toi", "TongDaUong", "ConLai"]

def lay_du_lieu_cu(username):
    if not os.path.exists(FILE_EXCEL): return None
    try:
        df = pd.read_excel(FILE_EXCEL)
        user_hist = df[df["Ten"] == username]
        if not user_hist.empty:
            last_row = user_hist.iloc[-1]
            return {"sex": last_row["GioiTinh"], "height": int(last_row["ChieuCao"]), "weight": int(last_row["CanNang"])}
    except: pass
    return None

def kiem_tra_hom_nay(username):
    ngay_hom_nay = datetime.now().strftime("%d/%m/%Y")
    if not os.path.exists(FILE_EXCEL): return False, 0, 0
    try:
        df = pd.read_excel(FILE_EXCEL)
        for col in COLUMNS:
            if col not in df.columns: df[col] = 0
        user_data = df[(df["Ten"] == username) & (df["Ngay"] == ngay_hom_nay)]
        if not user_data.empty:
            return True, int(user_data.iloc[0]["CanUong"]), int(user_data.iloc[0]["TongDaUong"])
    except: pass
    return False, 0, 0

def cap_nhat_excel(data_full, luong_vua_uong):
    ngay_hom_nay = datetime.now().strftime("%d/%m/%Y")
    username = data_full["name"]
    gio = datetime.now().hour
    cot_buoi = "Sang"
    if 11 <= gio < 14: cot_buoi = "Trua"
    elif 14 <= gio < 18: cot_buoi = "Chieu"
    elif gio >= 18 or gio < 5: cot_buoi = "Toi"

    if not os.path.exists(FILE_EXCEL): df = pd.DataFrame(columns=COLUMNS)
    else: df = pd.read_excel(FILE_EXCEL)
    for col in COLUMNS:
        if col not in df.columns: df[col] = 0

    condition = (df["Ten"] == username) & (df["Ngay"] == ngay_hom_nay)
    
    if df[condition].empty:
        new_row = {col: 0 for col in COLUMNS}
        new_row.update({
            "Ngay": ngay_hom_nay, "Ten": username, 
            "GioiTinh": data_full["sex"], "ChieuCao": data_full["height"], "CanNang": data_full["weight"], 
            "CanUong": data_full["total_need"], 
            cot_buoi: luong_vua_uong, "TongDaUong": luong_vua_uong, "ConLai": data_full["total_need"] - luong_vua_uong
        })
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df.loc[condition, cot_buoi] += luong_vua_uong
        df.loc[condition, "TongDaUong"] += luong_vua_uong
        df.loc[condition, "ConLai"] = df.loc[condition, "CanUong"] - df.loc[condition, "TongDaUong"]
        df.loc[df["ConLai"] < 0, "ConLai"] = 0
    
    try: 
        df.to_excel(FILE_EXCEL, index=False)
        print("-> Đã lưu Excel.")
    except: 
        noi("Lỗi lưu file Excel (Đang mở?).", wait=True)

def gui_len_iot(uong_lan_nay, tong_da_uong, tong_can, con_thieu):
    """Gửi dữ liệu uống nước của 1 lần lên IoT platform"""
    try:
        payload = {
            "device_id": DEVICE_ID,
            "api_key": DEVICE_API_KEY,
            "metric_type": "water_intake_ml",
            "value": float(uong_lan_nay),
            "payload": {
                "user": session_data.get("name"),
                "this_intake_ml": float(uong_lan_nay),
                "total_today_ml": float(tong_da_uong),
                "target_ml": float(tong_can),
                "remaining_ml": float(con_thieu)
            }
        }
        r = requests.post(f"{BACKEND_URL}/ingest/telemetry", json=payload, timeout=5)
        print(">>> GUI LEN IOT:", r.status_code, r.text)
    except Exception as e:
        print("Loi gui len IoT:", e)

# 4. CÁC GIAI ĐOẠN CHÍNH
def buoc_1_check_in():
    global db_encs, db_names 
    if not db_encs:
        nap_du_lieu()
        
    print("\n--- BƯỚC 1: CHECK-IN ---")
    cap = cv2.VideoCapture(0)
    face_start = time.time()
    user_found = None
    
    noi("Chào bạn. Mời nhìn vào camera để check in.", wait=False)

    while True:
        ret, frame = cap.read(); frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        box_size = 320
        sx, sy = (w-box_size)//2, (h-box_size)//2
        ex, ey = sx+box_size, sy+box_size
        nen = (frame.copy()*0.4).astype(np.uint8); nen[sy:ey, sx:ex] = frame[sy:ey, sx:ex]; frame = nen
        cv2.rectangle(frame, (sx, sy), (ex, ey), (255,0,0), 2)
        cv2.putText(frame, "DOCTOR.X ZONE", (sx, sy-10), 1, 1, (255,0,0), 2)

        if time.time() - face_start < 3:
            sec = int(3 - (time.time() - face_start)) + 1
            cv2.putText(frame, str(sec), (sx+140, sy+180), 1, 5, (0,255,255), 5)
        else:
            small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            if locs:
                encs = face_recognition.face_encodings(rgb, locs)
                matches = face_recognition.compare_faces(db_encs, encs[0])
                if True in matches:
                    user_found = db_names[matches.index(True)]
                    cv2.rectangle(frame, (sx, ey-50), (ex, ey), (0,255,0), -1)
                    cv2.putText(frame, user_found, (sx+20, ey-10), 1, 2, (255,255,255), 2)
                    cv2.rectangle(frame, (sx, sy), (ex, ey), (0,255,0), 3)
                    cv2.imshow('DOCTOR.X', frame); cv2.waitKey(1000)
                    cap.release(); cv2.destroyAllWindows()
                    return user_found
                else:
                    # --- XỬ LÝ NGƯỜI LẠ ---
                    cv2.rectangle(frame, (sx, sy), (ex, ey), (0,0,255), 3)
                    cv2.putText(frame, "NGUOI LA", (sx+50, ey-10), 1, 2, (0,0,255), 2)
                    cv2.imshow('DOCTOR.X', frame); cv2.waitKey(1000)
                    cap.release(); cv2.destroyAllWindows()
                    
                    noi("Bạn chưa có trong hệ thống. Nếu muốn đăng ký, hãy nói rõ: Tôi Đăng Ký. Nếu không, hãy nói: Tôi Không Đăng Ký.", wait=True)
                    traloi = None
                    while traloi is None:
                        traloi = nghe("dang_ky")
                        if traloi is None: noi("Mời nói lại: Tôi Đăng Ký hoặc Tôi Không Đăng Ký?", wait=True)
                    
                    if traloi == True:
                        noi("Mời bạn đọc to Họ và Tên của bạn?", wait=True)
                        ten_moi = None
                        while ten_moi is None:
                            ten_moi = nghe("ho_ten")
                            if ten_moi is None: noi("Tên quá ngắn. Mời đọc lại cả Họ và Tên.", wait=True)
                        
                        noi(f"Chào {ten_moi}. Mời nhìn vào camera để chụp ảnh.", wait=True)
                        
                        cap = cv2.VideoCapture(0)
                        for i in range(30): cap.read()
                        ret, new_frame = cap.read(); new_frame = cv2.flip(new_frame, 1)
                        if not os.path.exists(PATH_ANH): os.makedirs(PATH_ANH)
                        cv2.imwrite(f"{PATH_ANH}/{ten_moi}.jpg", new_frame)
                        cap.release()
                        noi("Đã lưu hình ảnh. Đang cập nhật dữ liệu...", wait=True)
                        nap_du_lieu()
                        return ten_moi
                    else:
                        return None
        
        cv2.imshow('DOCTOR.X', frame)
        if cv2.waitKey(1) == ord('q'): break
    
    cap.release(); cv2.destroyAllWindows()
    return None

def buoc_2_quet_chai():
    print("\n--- BƯỚC 2: QUÉT CHAI ---")
    noi(f"Chào {session_data['name']}. Đưa chai nước vào khung bên phải.", wait=True)
    cap = cv2.VideoCapture(0); counter = 0
    while True:
        ret, frame = cap.read(); frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        bx1, by1 = w-200, (h-350)//2; bx2, by2 = w-50, by1+350
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,255,255), 2)
        res = MODEL_YOLO(frame, verbose=False); found = False
        for r in res[0].boxes:
            if int(r.cls[0]) == 39:
                box = r.xyxy[0].int().tolist(); cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
                if bx1 < cx < bx2 and by1 < cy < by2:
                    found = True; cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,255,0), 3)
                    counter += 1; cv2.rectangle(frame, (bx1, by2+10), (bx1+int(150*(counter/15)), by2+20), (0,255,0), -1)
                    if counter >= 15:
                        cv2.imshow('DOCTOR.X', frame); cv2.waitKey(500); cap.release(); cv2.destroyAllWindows()
                        noi(f"Đã xác nhận chai {DUNG_TICH_THAT}ml.", wait=True)
                        return
        if not found: counter = 0
        cv2.imshow('DOCTOR.X', frame)
        if cv2.waitKey(1) == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def buoc_3_phong_van():
    print("\n--- BƯỚC 3: PHỎNG VẤN ---")
    co_du_lieu, tong_can, da_uong = kiem_tra_hom_nay(session_data["name"])
    if co_du_lieu:
        session_data["total_need"] = tong_can
        con_lai = tong_can - da_uong
        if con_lai < 0: con_lai = 0
        noi(f"Chào mừng quay lại. Hôm nay bạn đã uống {da_uong}ml.", wait=True)
        noi(f"Mục tiêu còn lại là {con_lai}ml nữa.", wait=True)
        return 
    else:
        old_data = lay_du_lieu_cu(session_data["name"])
        can_cap_nhat = True
        if old_data:
            noi(f"Dữ liệu cũ: {old_data['sex']}, {old_data['height']}cm, {old_data['weight']}kg.", wait=True)
            noi("Bạn có muốn cập nhật lại không?", wait=True)
            traloi = None
            while traloi is None:
                traloi = nghe("dang_ky") 
                if traloi is None: noi("Mời nói rõ: Tôi Muốn hoặc Tôi Không Muốn?", wait=True)
            if traloi == False: 
                session_data["sex"] = old_data["sex"]; session_data["height"] = old_data["height"]; session_data["weight"] = old_data["weight"]
                can_cap_nhat = False
                noi("Đã ghi nhận dữ liệu cũ.", wait=True)

        if can_cap_nhat:
            while not session_data["sex"]: noi("Bạn là đàn ông hay phụ nữ?", wait=True); session_data["sex"] = nghe("sex")
            while not session_data["height"]: noi("Bạn cao bao nhiêu?", wait=True); session_data["height"] = nghe("so")
            while not session_data["weight"]: noi("Bạn nặng bao nhiêu ký?", wait=True); session_data["weight"] = nghe("so")

        s, h, w = session_data["sex"], session_data["height"], session_data["weight"]
        factor = 40 if s == "Nam" else 35
        total = w * factor + (200 if h > 175 else 0)
        session_data["total_need"] = total
        so_chai = round(total / DUNG_TICH_THAT, 1)
        tb_buoi = int(total / 4)
        noi(f"Kết quả: Bạn cần uống {total}ml mỗi ngày.", wait=True)
        noi(f"Tương đương {so_chai} chai. Mỗi buổi khoảng {tb_buoi}ml.", wait=True)

def buoc_4_moi_uong():
    print("\n--- BƯỚC 4: UỐNG NƯỚC ---")
    noi("Mời bạn uống nước. Uống xong bấm phím Enter.", wait=True)
    input(">> BẤM ENTER ĐỂ TIẾP TỤC...") 

def buoc_5_kiem_tra_lai():
    print("\n--- BƯỚC 5: KIỂM TRA LẠI ---")
    noi("Đưa chai đã uống vào camera.", wait=False)
    cap = cv2.VideoCapture(0); counter = 0
    while True:
        ret, frame = cap.read(); frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        bx1, by1 = w-200, (h-350)//2; bx2, by2 = w-50, by1+350
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,255,255), 2)
        res = MODEL_YOLO(frame, verbose=False); found = False
        for r in res[0].boxes:
            if int(r.cls[0]) == 39:
                box = r.xyxy[0].int().tolist(); cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
                if bx1 < cx < bx2 and by1 < cy < by2:
                    found = True; cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,255,0), 3)
                    counter += 1; cv2.rectangle(frame, (bx1, by2+10), (bx1+int(150*(counter/15)), by2+20), (0,255,0), -1)
                    if counter >= 15:
                        cv2.imshow('DOCTOR.X', frame); cv2.waitKey(500); cap.release(); cv2.destroyAllWindows()
                        noi("Tôi thấy chai. Bạn còn lại khoảng bao nhiêu mi li lít?", wait=True)
                        con_lai = None
                        while con_lai is None:
                            con_lai = nghe("so")
                            if con_lai is None: noi("Ví dụ hãy nói: 50.", wait=True)
                            elif con_lai > DUNG_TICH_THAT: noi("Vô lý. Mời nói lại.", wait=True); con_lai = None
                        uong_lan_nay = DUNG_TICH_THAT - con_lai
                        cap_nhat_excel(session_data, uong_lan_nay)
                        _, tong_can, tong_da_uong = kiem_tra_hom_nay(session_data["name"])
                        con_thieu = tong_can - tong_da_uong
                        gui_len_iot(uong_lan_nay, tong_da_uong, tong_can, con_thieu)
                        chai_thieu = round(con_thieu / DUNG_TICH_THAT, 1)
                        noi(f"Ghi nhận uống {uong_lan_nay}ml.", wait=True)
                        noi(f"Tổng hôm nay: {tong_da_uong}ml. Còn thiếu {con_thieu}ml.", wait=True)
                        noi(f"Cố lên! Còn khoảng {chai_thieu} chai.", wait=True)
                        return
        if not found: counter = 0
        cv2.imshow('DOCTOR.X', frame)
        if cv2.waitKey(1) == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# MAIN
if __name__ == "__main__":
    nap_du_lieu()
    intro = "Chào bạn, tôi tên là Doctor X, tôi sẽ là người giám sát việc uống nước của bạn. Mời bạn check in"
    noi(intro, wait=True)
    user = buoc_1_check_in()
    if user:
        session_data["name"] = user
        buoc_2_quet_chai()
        buoc_3_phong_van()   
        buoc_4_moi_uong()
        buoc_5_kiem_tra_lai() 
    else:
        noi("Rất tiếc, bạn chọn không đăng ký. Hẹn gặp lại!", wait=True)
    
    print("\n--- KẾT THÚC ---")