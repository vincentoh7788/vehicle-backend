# Vehicle Counting Application Backend using Deep Learning
The Deep Learning Model used is YOLOv8. The model is trained with 1000+ data within Malaysian traffic dataset. https://app.roboflow.com/project-ch1aj/vehicle-detection-swchc/8
The backend is created in REST API using Flask
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
1. YOLO -> To run the trained YOLOv8 for inferencing
2. flask -> To get request, make response and send files
3. flask_cors -> Enable CORS (Cross-origin resource sharing)
4. flask_jwt_extended -> JWT Settings
5. flask_bcrypt -> Encrypt the user password
6. deep_sort -> Vehicle tracking
7. cv2 -> Image processing
8. psycopg2 -> PostgreSQL database adapter for the Python programming language
9. torch -> To run YOLO model which is based on PyTorch
10. tensorflow -> Requires for the DeepSORT
