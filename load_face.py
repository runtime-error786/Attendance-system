import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pandas as pd

def load_known_faces(images_dir, names_list):
    known_faces = []
    known_names = names_list
    
    for filename in os.listdir(images_dir):
        if filename == '.DS_Store':  
            continue
        image = face_recognition.load_image_file(f"{images_dir}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
    
    return known_faces, known_names