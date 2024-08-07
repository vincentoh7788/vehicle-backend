from flask import Flask, request, jsonify,Response,send_from_directory,send_file
from flask_cors import CORS
from ultralytics import YOLO
from flask_bcrypt import Bcrypt
import os
from flask_jwt_extended import create_access_token,get_jwt,get_jwt_identity,unset_jwt_cookies, jwt_required, JWTManager,create_refresh_token
import psycopg2
import json
import pytz
import numpy as np
from datetime import datetime,timedelta,timezone
import cv2
import torch
from collections import deque
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet
from helper import create_video_writer
app = Flask(__name__)
bcrypt = Bcrypt(app) 
CORS(app,origins="*")
jwt = JWTManager(app)
UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
OUTPUT_FOLDER = './output_video'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['JWT_SECRET_KEY'] = '******'
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
def get_db_connection():
    #Localhost
    conn = psycopg2.connect(host='localhost',
                            database='*******',
                            user='postgres',
                            password='******')

    return conn

#Prevent preflight request
@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        response = Response()
        return response
    
#Prevent preflight request    
@app.route('/getname', methods=['OPTIONS'])
def getname_options():
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    return response 

#Prevent preflight request 
@app.route('/getresult', methods=['OPTIONS'])
def getresult_options():
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    return response
  
@app.route('/')
def home():
    return 'This is just a backend'

#Main function 
@app.route('/output',methods = ['POST'])
def counting_result():
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        torch.cuda.set_device(0)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
    # Initialize the video capture and the video writer objects
    video_cap = cv2.VideoCapture(file_path)
    date_name = datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
    output_file = date_name.strftime('%Y%m%d_%H%M%S') + '_' + filename
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
    writer = create_video_writer(video_cap,output_path)
    width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    line_y = int(request.form.get('lineHeight')) 
    print(line_y)
    # define some parameters
    conf_threshold = 0.5
    max_cosine_distance = 0.4
    nn_budget = None
    points = [deque(maxlen=32) for _ in range(1000)] # list of deques to store the points
    counter_in = [0,0,0,0,0,0]
    counter_out = [0,0,0,0,0,0]
    #line_y = int(height*2/3)
    start_line_A = (0, line_y)
    end_line_A = (int(width), line_y)

    # Initialize the YOLOv8 model using the default weights
    model = YOLO("best_v8_pro.pt")

    # Initialize the deep sort tracker
    model_filename = "config/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # load the COCO class labels the YOLO model was trained on
    classes_path = "config/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # create a list of random colors to represent each class
    np.random.seed(42)  # to get the same colors
    colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

    # loop over the frames
    while True:
    # starter time to computer the fps
        start = datetime.now()
        ret, frame = video_cap.read()
        if frame is None:
            break
        overlay = frame.copy()
        # draw the lines
        cv2.line(frame, start_line_A, end_line_A, (0, 255, 0), 12)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        # if there is no frame, we have reached the end of the video
        if not ret:
            print("End of the video file...")
            break
    # run the YOLO model on the frame
        results = model(frame,rect=True,imgsz=[576,384])

    # loop over the results
        for result in results:
            # initialize the list of bounding boxes, confidences, and class IDs
            bboxes = []
            confidences = []
            class_ids = []

        # loop over the detections
            for data in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = data
                x = int(x1)
                y = int(y1)
                w = int(x2) - int(x1)
                h = int(y2) - int(y1)
                class_id = int(class_id)
                # filter out weak predictions by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > conf_threshold:
                    bboxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
    ############################################################
    ### Track the objects in the frame using DeepSort        ###
    ############################################################

        # get the names of the detected objects
        names = [class_names[class_id] for class_id in class_ids]

        # get the features of the detected objects
        features = encoder(frame, bboxes)
        # convert the detections to deep sort format
        dets = []
        for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
            dets.append(Detection(bbox, conf, class_name, feature))

        # run the tracker on the detections
        tracker.predict()
        tracker.update(dets)

        # loop over the tracked objects
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # get the bounding box of the object, the name
            # of the object, and the track id
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_name = track.get_class()
            # convert the bounding box to integers
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # get the color associated with the class name
            class_id = class_names.index(class_name)
            color = colors[class_id]
            B, G, R = int(color[0]), int(color[1]), int(color[2])

        # draw the bounding box of the object, the name
        # of the predicted object, and the track id
            text = str(track_id) + " - " + class_name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 3)
            cv2.rectangle(frame, (x1 - 1, y1 - 20),
                      (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        ############################################################
        ### Count the number of vehicles passing the lines       ###
        ############################################################
        
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            # append the center point of the current object to the points list
            points[track_id].append((center_x, center_y))

            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
        
        # loop over the set of tracked points and draw them
            for i in range(1, len(points[track_id])):
                point1 = points[track_id][i - 1]
                point2 = points[track_id][i]
            # if the previous point or the current point is None, do nothing
                if point1 is None or point2 is None:
                    continue
            
                cv2.line(frame, (point1), (point2), (0, 255, 0), 2)
            
            # get the last point from the points list and draw it
            last_point_x = points[track_id][0][0]
            last_point_y = points[track_id][0][1]
            cv2.circle(frame, (int(last_point_x), int(last_point_y)), 4, (255, 0, 255), -1)    

        # if the y coordinate of the center point is below the line, and the x coordinate is 
        # between the start and end points of the line, and the the last point is above the line,
        # increment the total number of cars crossing the line and remove the center points from the list
            for i in range(6):
                if center_y > start_line_A[1]  and last_point_y < start_line_A[1] and i == class_id:
                    counter_in[i] += 1
                    points[track_id].clear()
                elif center_y < start_line_A[1]  and last_point_y > start_line_A[1] and i == class_id:
                    counter_out[i] += 1
                    points[track_id].clear()
        #elif center_y > start_line_C[1] and start_line_C[0] < center_x < end_line_C[0] and last_point_y < start_line_A[1]:
            #counter_C += 1
            #points[track_id].clear()
            
    ############################################################
    ### Some post-processing to display the results          ###
    ############################################################

        # end time to compute the fps
        end = datetime.now()
        # calculate the frame per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255),6)
    
    # draw the total number of vehicles passing the lines
        for i in range(6):
            cv2.putText(frame, f"{class_names[i]} in: {counter_in[i]}", (50,100+ i*50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, f"{class_names[i]} out: {counter_out[i]}", (int(width/2),100+ i*50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)
    #cv2.putText(frame, "C", (910, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #cv2.putText(frame, f"{counter_in}", (60,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)
    #cv2.putText(frame, f"{counter_out}", (60,90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)
    #cv2.putText(frame, f"{counter_C}", (1040, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # write the frame to disk
        #cv2.imshow("Output", frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord("q"):
            break

# release the video capture, video writer, and close all windows
    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return jsonify({'counter_in': counter_in, 'counter_out': counter_out, 'output_video': output_file})

#Preview video
@app.route('/getvideo', methods=['GET'])
def display_video():
    try:
        filename = request.args.get('filename')
        if filename:
            return send_from_directory(app.config['OUTPUT_FOLDER'], filename,as_attachment=False)
        else:
            return "Filename parameter is missing", 400 
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#Download video
@app.route('/downloadvideo', methods=['GET'])
def download_video():
    filename = request.args.get('filename')
    if filename:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    else:
        return "Filename parameter is missing", 400 

 #Save result to database       
@app.route('/save',methods=['POST'])
@jwt_required()
def save_result():
    useremail = get_jwt_identity()
    counter_in = request.json['counter_in']
    counter_out = request.json['counter_out']
    output_video = request.json['output_video']
    now = datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO result_table (video_link, email, curr_time,counter_in, counter_out) VALUES (%s, %s, %s, %s, %s)', 
        (output_video, useremail, now.strftime('%Y-%m-%d %H:%M:%S'), counter_in, counter_out,))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'message': 'Result saved'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#Get result from database
@app.route('/getresult',methods=['GET'])
@jwt_required()
def get_result():
    try:
        useremail = get_jwt_identity()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT result_id, video_link, curr_time, counter_in, counter_out FROM result_table WHERE email = %s', (useremail,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500   

@app.route('/deleteresult', methods=['DELETE'])
@jwt_required()
def delete_result():
    id = request.args.get('id')
    if id is None:
        return jsonify({'error': 'No id provided'}), 400
    try:
        id = int(id)
    except ValueError:
        return jsonify({'error': 'Invalid id'}), 400
    try:
        useremail = get_jwt_identity()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT video_link FROM result_table WHERE result_id = %s AND email = %s', (int(id), useremail))
        video_link = cur.fetchone()
        if video_link is None:
            return jsonify({'error': 'No video found for this id'}), 404
        
        video_link = video_link[0]
        video_file_path = os.path.join(app.config['OUTPUT_FOLDER'],video_link)
        if os.path.exists(video_file_path):
            os.remove(video_file_path)
        
        cur.execute('DELETE FROM result_table WHERE result_id = %s AND email = %s', (int(id), useremail))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'message': 'Result deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
#Signup
@app.route('/signup',methods=['POST'])
def user_signup():
    try:
        if request.method == 'POST':
            useremail = request.json['email']
            username = request.json['username']
            userpassword = request.json['password']
            hashed_password = bcrypt.generate_password_hash(userpassword).decode('utf8')
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('INSERT INTO app_user_table (email,password,username) VALUES (%s, %s, %s)'
                        ,(useremail,hashed_password,username,))
            conn.commit()
            cur.close()
            conn.close()
        return jsonify({'message': 'User signed up successfully', 'email': useremail}), 201
    
    except Exception as e:
            return jsonify({'error': str(e)}), 500

#Login and establish JWT
@app.route("/login",methods=['POST'])
def user_login():
    try:
        if request.method == 'POST':
            useremail = request.json['email']
            userpassword = request.json['password']
            conn = get_db_connection()
            cur = conn.cursor()
            #Handle authentication
            cur.execute("SELECT password FROM app_user_table WHERE email = %s", (useremail,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result:
                hashed_password = result[0]
                if bcrypt.check_password_hash(hashed_password,userpassword):
                    access_token = create_access_token(identity=useremail)
                    response = jsonify({'token':access_token})
                    return response,200
                else:
                    return jsonify({'message': 'wrong password'}), 401
            else:
                return jsonify({'error': 'Invalid email or password'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#Refresh JWT    
@app.after_request
def refresh_expiring_jwts(response):
    try:
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
        if target_timestamp > exp_timestamp:
            access_token = create_access_token(identity=get_jwt_identity())
            data = response.get_json()
            if type(data) is dict:
                print(data)
                data["access_token"] = access_token 
                response.data = json.dumps(data)
        return response
    except (RuntimeError, KeyError):
        # Case where there is not a valid JWT. Just return the original response
        return response

 #Handle logout      
@app.route("/logout", methods=["POST"])
def logout():
    response = jsonify({"msg": "logout successful"})
    unset_jwt_cookies(response)
    return response

#Get username(Check status JWT)   
@app.route("/getname", methods=["GET"])
@jwt_required()
def getUsername():
    useremail = get_jwt_identity()
    conn = get_db_connection()
    cur = conn.cursor()
    #Get username
    cur.execute("SELECT username FROM app_user_table WHERE email = %s", (useremail,))
    username = cur.fetchone()[0]
    cur.close()
    conn.close()
    response = jsonify({"username":username})
    return response

if __name__ == '__main__':
        app.run()