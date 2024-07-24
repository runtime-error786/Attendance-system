import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pandas as pd
from load_face import load_known_faces
from attendance import record_attendance
from process_video import process_video_stream

if __name__ == "__main__":
    names_list = ["Mustafa"] 
    images_dir = r"C:\Users\musta\OneDrive\Desktop\face recognition\img"  
    
    known_faces, known_names = load_known_faces(images_dir, names_list)
    process_video_stream(known_faces, known_names)
