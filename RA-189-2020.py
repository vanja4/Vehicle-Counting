import sys
import pandas as pd
from matplotlib import axes
import numpy as np
import cv2
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

folder_path = sys.argv[1]
predicted_counts= []

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

train_dir = folder_path + 'pictures/'

df = pd.read_csv("data1/counts.csv")
true_counts = df["Broj_prelaza"].values

pos_imgs = []
neg_imgs = []
for img_name in os.listdir(train_dir):
        img_path = os.path.join(train_dir, img_name)
        img = load_image(img_path)
        if 'p_' in img_name:
            pos_imgs.append(img)
        elif 'n_' in img_name:
            neg_imgs.append(img)

def get_hog():
    img = load_image('data1/pictures/p_1.png')
    nbins = 9 
    cell_size = (8, 8)  
    block_size = (2, 2)

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog

def train_classifier(hog_descriptor):
    pos_features = []
    neg_features = []
    labels = []
    
    for img in pos_imgs:
        pos_features.append(hog_descriptor.compute(img))
        labels.append(1)

    for img in neg_imgs:
        neg_features.append(hog_descriptor.compute(img))
        labels.append(0)

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    x = np.vstack((pos_features, neg_features))
    y = np.array(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    classifier = LinearSVC()
    classifier.fit(x_train, y_train)
    
    return classifier

def detect_line(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, 55, 165, apertureSize=3)

    min_line_length = 200
    
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)

    x1 = 0
    y1 = -595
    x2 = 3750
    y2 = -595

    return (x1, y1, x2, y2)
    
def get_line_params(line_coords):
    k = (float(line_coords[3]) - float(line_coords[1])) / (float(line_coords[2]) - float(line_coords[0]))
    n = k * (float(-line_coords[0])) + float(line_coords[1])
    return k, n
    
def detect_cross(x, y, k, n):
    
    yy = k*x + n
    
    return 20 <= yy - y <= 100

def process_video(video_path, hog_descriptor, classifier):
    sum_of_vehicles = 0
    k = 0
    n = 0

    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) 
    detected_vehicles = set() 
  
    while True:
        frame_num += 1
        grabbed, frame = cap.read()

        if not grabbed:
            break

        frame_cropp = frame[610:1395, 1420:2460]
        
        if frame_num == 1: 
            line_coords = detect_line(frame)
           
            k, n = get_line_params(line_coords)

            line_left_x = line_coords[0]
            line_right_x = line_coords[2]
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh, frame_bin = cv2.threshold(frame_gray, 130, 255, cv2.THRESH_BINARY)
        frame_contours = cv2.dilate(frame_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
        contours, _ = cv2.findContours(frame_contours.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(contour) for contour in contours]

        for rectangle in rectangles:
            x, y, w, h = rectangle
            area = w * h
            if h > 79 and w > 20 and area < 16000: 
                
                roi = frame_contours[y:y+h, x:x+w]
                roi = cv2.resize(roi, (60, 120), interpolation=cv2.INTER_NEAREST)

                vehicles_features = hog_descriptor.compute(roi).reshape(1, -1)
                predicted_number = classifier.predict(vehicles_features)[0]

                center_x = x + w / 2
                center_y = 480 - (y + h / 2)
                
                if (line_left_x <= center_x <= line_right_x) and (detect_cross(center_x, center_y, k, n)):
                    
                    vehicle_id = (int(center_x // 30), int(center_y // 30))

                    if vehicle_id not in detected_vehicles:
                        detected_vehicles.add(vehicle_id)
                        sum_of_vehicles += predicted_number

                   
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                    
                    cv2.imshow("Frame", frame_cropp)
                    cv2.waitKey(2)
            
    cap.release()
    cv2.destroyAllWindows()
    return sum_of_vehicles

def process_all_videos(folder_path, hog_descriptor, classifier):
    global predicted_counts
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        suma = process_video(video_path, hog_descriptor, classifier)
        name = video_file.split("\\")[-1].split(".")[0]
        predicted_counts = np.append(predicted_counts, suma)
        
        print(f"{name}-{true_counts[video_files.index(video_file)]}-{suma}")

hog1 = get_hog()
classifier1 = train_classifier(hog1)

true_counts = np.array(true_counts).astype(np.int32)

process_all_videos(folder_path + "videos", hog1, classifier1)
mae = mean_absolute_error(true_counts, predicted_counts)
print("MAE:", mae)