import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pandas as pd


def record_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    if not os.path.isfile('attendance.xlsx'):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        df.to_excel('attendance.xlsx', index=False)
    
    df = pd.read_excel('attendance.xlsx')
    
    new_entry = pd.DataFrame([{'Name': name, 'Date': date_str, 'Time': time_str}])
    
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_excel('attendance.xlsx', index=False)