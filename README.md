Flask: Web framework to create routes and serve pages.

cv2 (OpenCV): To capture webcam feed and manipulate images.

mediapipe: Hand tracking library.

numpy: Array operations.

math: For speed, distance, and angular velocity calculations.

deque: Efficient fixed-length queues to store movement history.

Tracks 1 hand with high detection/tracking confidence.

mp_draw is used to draw landmarks and connections on the hand.

Starts webcam capture (default camera = 0).

Later, you can switch camera using the Flask form.

canvas stores the drawing separately from webcam feed.

prev_x, prev_y track last finger position.

mode switches between Draw and Erase.

latest_frame stores the current frame for saving.

Tracks trajectory, speed, acceleration, angular velocity, distance.

deque ensures only the latest N points are stored.

Returns list of fingers up (1) or down (0).

Thumb uses x-axis, others use y-axis.

7. Video Frame Generation (generate_frames)

This is the heart of the app:

Capture frame & flip (mirror view).

Convert BGR → RGB for MediaPipe processing.

Detect hand landmarks.

Finger gestures:

Two fingers (index + middle): Red

Index + ring: Green

Index + pinky: Blue

Middle + ring: Black

No fingers: Erase

Draw / Erase based on finger positions.

Scientific calculations:

Speed = distance moved per frame.

Acceleration = change in speed.

Distance traveled.

Angular velocity = change in angle of movement.

Blend canvas with live webcam feed.

Add toolbar for color selection and mode display.

Add scientific panel with graphs.

Return frame as JPEG bytes for Flask streaming.

Open browser at http://localhost:5000.

Draw gestures:

Index finger alone → draw.

No fingers → erase.

Finger combinations → select color.

View scientific panel:

Speed, acceleration, angular velocity, distance.

Mini XY trajectory plot.

Switch cameras from dropdown if multiple cameras are available.

Save drawing by clicking the “Save Drawing” button.

 Key Features

Real-time hand-tracking drawing.

Gesture-based color selection (red, green, blue, black).

Eraser mode via closed fist.

Scientific panel with:

X/Y coordinates.

Speed, acceleration, angular velocity.

Distance traveled.

Graphs and mini trajectory plot.

Save drawing directly as PNG.

Switch between multiple cameras without restarting.
