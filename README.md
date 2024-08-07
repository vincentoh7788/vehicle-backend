# Vehicle Counting Application Backend using Deep Learning
The model YOLOv8 is chosen among 4 models to be deployed within the application. The model is trained with 1000+ data within Malaysian traffic dataset. https://app.roboflow.com/project-ch1aj/vehicle-detection-swchc/8
<br> The backend is created in REST API using Flask
<br>
Model evaluation:
1. Precision = 0.873
2. Recall = 0.844
3. F1-score = 0.858
4. mAP50 = 0.885
5. mAP50-95 = 0.658
6. FPS = 93
<br>
Training and evaluation: https://colab.research.google.com/drive/1timgSs15BAyOJth8smh-p8fQmV0gz-IP#scrollTo=Bsc0DSA9nMqj

# Function of API
1. Make Prediction on video
   YOLOv8 model makes prediction on each frame of the video.
   DeepSORT predicts the location based on previous frame.
   Counting algorithm will count the number of each type of vehicle based on a line.
2. Basic functionality such as Create/Login/Logout account
   Use JWT to handle the user session
3. Preview and download output video
4. Track previous result
# Main package
**View the package version at requirements.txt**
1. ultralytics -> To run the trained YOLOv8 for inferencing (https://github.com/ultralytics/ultralytics)
2. flask -> To get request, make response and send files (https://github.com/pallets/flask)
3. flask_cors -> Enable Cross-origin resource sharing (https://github.com/corydolphin/flask-cors)
4. flask_jwt_extended -> JWT Settings (https://github.com/vimalloc/flask-jwt-extended)
5. flask_bcrypt -> Encrypt the user password (https://github.com/maxcountryman/flask-bcrypt)
6. deep_sort -> Vehicle tracking (https://github.com/nwojke/deep_sort)
7. cv2 (OpenCV) -> Image processing (https://github.com/opencv/opencv)
8. psycopg2 -> PostgreSQL database adapter for the Python programming language (https://github.com/psycopg/psycopg2)
9. torch -> To run YOLO model which is based on PyTorch (https://github.com/pytorch/pytorch)
10. tensorflow -> Requires for the DeepSORT (https://github.com/tensorflow/tensorflow)



