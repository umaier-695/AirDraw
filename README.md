# Air Draw – Scientific Mode

**Air Draw – Scientific Mode** is an interactive Python Flask application that enables real-time hand-tracking-based drawing using your webcam. Leveraging **MediaPipe Hands** and **OpenCV**, it allows users to draw, erase, and select colors with simple hand gestures, providing a touch-free creative experience. The application includes a **scientific panel** that displays real-time metrics such as speed, acceleration, angular velocity, and distance traveled, and it plots these parameters on dynamic graphs alongside a mini coordinate system. Users can switch cameras on-the-fly and save their drawings as images. This repository also includes sample screenshots and a test video demonstrating the functionality of the app. Air Draw is designed for educational, experimental, and interactive purposes, making it ideal for exploring gesture recognition, computer vision, and scientific visualization in Python.

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/umaier-695/AirDraw.git
cd AirDraw
Create and activate a virtual environment (optional but recommended)

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
Install required Python packages

bash
Copy code
pip install -r requirements.txt
If requirements.txt is not present, install manually:

bash
Copy code
pip install flask opencv-python mediapipe numpy
Usage

Run the Flask application

python Air_draw.py


Open your web browser and go to:

http://127.0.0.1:5000/


Features

Draw or erase using hand gestures

Select colors using gestures: red, green, blue, black

Switch between multiple cameras

View real-time scientific metrics: speed, acceleration, angular velocity, distance

Save drawings as PNG images
