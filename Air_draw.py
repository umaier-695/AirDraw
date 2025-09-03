from flask import Flask, Response, render_template_string, request, send_file, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import io

app = Flask(__name__)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start with default camera
cap_index = 0
cap = cv2.VideoCapture(cap_index)

canvas = None
prev_x, prev_y = 0, 0
brush_thickness = 5
eraser_thickness = 50
draw_color = (0, 0, 255)
mode = "Draw"
latest_frame = None

# Scientific data
trajectory = deque(maxlen=50)
speed_history = deque(maxlen=100)
ang_velocity_history = deque(maxlen=100)
accel_history = deque(maxlen=100)
distance_traveled = 0
prev_point = None
prev_angle = None
speed = 0
ang_velocity = 0
acceleration = 0

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = [1 if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x else 0]
    for i in range(1,5):
        fingers.append(1 if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y else 0)
    return fingers

def draw_graph(panel, data, max_val, start_x, start_y, label, color=(0,255,0)):
    length = len(data)
    for i in range(1,length):
        y1 = int(start_y - (data[i-1]/max_val*40))
        y2 = int(start_y - (data[i]/max_val*40))
        cv2.line(panel, (start_x+i, y1), (start_x+i+1, y2), color, 2)
    cv2.putText(panel, label, (start_x, start_y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def draw_coordinate_system(panel, traj, width, height):
    grid_height = 80
    grid_top = height - grid_height - 10
    grid_left = 10
    grid_right = width - 10
    grid_bottom = height - 10
    cv2.rectangle(panel, (grid_left, grid_top), (grid_right, grid_bottom), (50,50,50), 1)
    for i in range(1,5):
        y = grid_top + i*(grid_height//5)
        cv2.line(panel, (grid_left, y), (grid_right, y), (50,50,50), 1)
    for i in range(1,5):
        x = grid_left + i*((grid_right-grid_left)//5)
        cv2.line(panel, (x, grid_top), (x, grid_bottom), (50,50,50), 1)
    for tx, ty in traj:
        plot_x = int(grid_left + (tx % (grid_right - grid_left)))
        plot_y = int(grid_top + ((ty % grid_height)))
        cv2.circle(panel, (plot_x, plot_y), 3, (0,255,255), -1)

def generate_frames():
    global canvas, prev_x, prev_y, draw_color, mode
    global trajectory, prev_point, prev_angle, speed, ang_velocity, acceleration, distance_traveled
    global speed_history, ang_velocity_history, accel_history, latest_frame, cap

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            finger_state = fingers_up(hand_landmarks)
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Color selection gestures
            if finger_state[1]==1 and finger_state[2]==1 and sum(finger_state)==2:
                draw_color = (0,0,255)
            elif finger_state[1]==1 and finger_state[3]==1 and sum(finger_state)==2:
                draw_color = (0,255,0)
            elif finger_state[1]==1 and finger_state[4]==1 and sum(finger_state)==2:
                draw_color = (255,0,0)
            elif finger_state[2]==1 and finger_state[3]==1 and sum(finger_state)==2:
                draw_color = (0,0,0)  # black

            # Drawing or erasing
            if sum(finger_state)==0:
                mode = "Erase"
                cv2.circle(frame, (x,y), 30, (0,0,0), 2)
                cv2.circle(canvas, (x,y), eraser_thickness, (0,0,0), -1)
            elif finger_state[1]==1 and sum(finger_state)==1:
                mode = "Draw"
                if prev_x==0 and prev_y==0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness, cv2.LINE_AA)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0,0

            # Scientific calculations
            trajectory.append((x,y))
            if prev_point:
                dx = x-prev_point[0]
                dy = y-prev_point[1]
                speed = math.sqrt(dx**2 + dy**2)
                acceleration = speed - speed_history[-1] if speed_history else 0
                distance_traveled += math.sqrt(dx**2 + dy**2)
                angle = math.atan2(dy, dx)
                if prev_angle is not None:
                    ang_velocity = math.degrees(angle-prev_angle)
                prev_angle = angle
            prev_point = (x,y)
            speed_history.append(speed)
            ang_velocity_history.append(ang_velocity)
            accel_history.append(acceleration)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(100,255,100), thickness=1))

        # Blend canvas with frame
        blur_canvas = cv2.GaussianBlur(canvas, (5,5),0)
        mask = cv2.cvtColor(blur_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask,20,255,cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        drawing = cv2.addWeighted(frame_bg,1,blur_canvas,1,0)

        # Toolbar with black visible and selected highlight
        cv2.rectangle(drawing, (0,0), (w,60), (50,50,50), -1)
        colors=[(0,0,255),(0,255,0),(255,0,0),(0,0,0)]
        for i,col in enumerate(colors):
            cv2.circle(drawing,(60+i*80,30),20,col,-1)
            cv2.circle(drawing,(60+i*80,30),24,(255,255,255),2)
            if col==draw_color:
                cv2.circle(drawing,(60+i*80,30),28,(0,255,255),3)
        cv2.putText(drawing,f"Mode: {mode}",(w-200,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        # Scientific panel
        panel_width = 200
        panel = np.zeros((h, panel_width,3), dtype=np.uint8)
        panel[:,:] = (30,30,30)
        top_y = 20
        cv2.putText(panel,"Scientific Panel",(10,top_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(panel,f"X: {prev_point[0] if prev_point else 0}",(10,top_y+30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(panel,f"Y: {prev_point[1] if prev_point else 0}",(10,top_y+50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(panel,f"Speed: {speed:.2f}",(10,top_y+70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(panel,f"Acceleration: {acceleration:.2f}",(10,top_y+90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(panel,f"Distance: {distance_traveled:.2f}",(10,top_y+110),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(panel,f"Angular Vel.: {ang_velocity:.2f}",(10,top_y+130),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        graph_start_y = h - 150
        graph_spacing = 50
        if len(speed_history)>1:
            draw_graph(panel, list(speed_history), max(speed_history)+1, 10, graph_start_y, "Speed", color=(0,255,0))
        if len(accel_history)>1:
            draw_graph(panel, list(accel_history), max([abs(a) for a in accel_history])+1, 10, graph_start_y-graph_spacing, "Accel", color=(0,200,255))
        if len(ang_velocity_history)>1:
            draw_graph(panel, list(ang_velocity_history), max([abs(v) for v in ang_velocity_history])+1, 110, graph_start_y, "Angular Vel", color=(255,100,0))

        draw_coordinate_system(panel, trajectory, panel_width, h)

        combined = np.hstack((drawing[:, :-panel_width], panel))
        latest_frame = combined.copy()

        ret, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Air Draw Scientific</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {background:black; margin:0; padding:0; font-family:sans-serif;}
            img {width:100%; height:auto; border:1px solid white;}
            h2 {color:white; text-align:center;}
            form {text-align:center; margin:10px;}
        </style>
    </head>
    <body>
        <h2>Air Draw - Scientific Mode</h2>
        <form method="POST" action="{{ url_for('change_camera') }}">
            <label style="color:white;">Select Camera:</label>
            <select name="camera_index">
                {% for i in range(5) %}
                <option value="{{ i }}" {% if i==0 %}selected{% endif %}>Camera {{ i }}</option>
                {% endfor %}
            </select>
            <button type="submit">Switch</button>
        </form>
        <div style="text-align:center;">
            <img src="{{ url_for('video_feed') }}">
        </div>
        <form method="POST" action="{{ url_for('save_image') }}">
            <button type="submit" style="font-size:20px;padding:10px 20px; margin:10px;">Save Drawing</button>
        </form>
    </body>
    </html>
    """)

@app.route('/change_camera', methods=['POST'])
def change_camera():
    global cap, cap_index
    new_index = int(request.form.get('camera_index', 0))
    if new_index != cap_index:
        cap.release()
        cap = cv2.VideoCapture(new_index)
        cap_index = new_index
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_image', methods=['POST'])
def save_image():
    global latest_frame
    if latest_frame is not None:
        _, buffer = cv2.imencode('.png', latest_frame)
        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        return send_file(io_buf, mimetype='image/png', as_attachment=True, download_name='AirDraw_Scientific.png')
    return redirect(url_for('index'))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
