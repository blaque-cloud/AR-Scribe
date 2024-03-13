import math
import random
import cv2
import keyboard
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, redirect, url_for, session, jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
lrn_toggle = False
accuracy = 0
ch = {}
k = 0
char = ''
ra = 0
cha = []
raa = []

frame_shape = (1080, 1920, 3)
imgCanvas = np.zeros((1080, 1920, 3), dtype='uint8')
AlphaMODEL = load_model("./static/models/alpha.h5")
AlphaLABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ''}


def prac():
    global cap

    prac_char = chr(random.randint(ord('A'), ord('Z')))

    cap = cv2.VideoCapture(0)
    video_capture2 = cv2.VideoCapture(f'.\\static\\char\\{prac_char}.mp4')

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(max_num_hands=1)

    draw = mp.solutions.drawing_utils

    colour = (1, 1, 1)
    thickness = 63
    font_size = 900
    font = ImageFont.truetype("./static/font/font.otf", font_size)

    mask = np.zeros(frame_shape, dtype='uint8')
    img_pil = Image.fromarray(mask)
    dra = ImageDraw.Draw(img_pil)
    dra.text((1000, 160), prac_char, font=font, fill=(1, 1, 1))
    mask = np.array(img_pil)

    prevxy = None

    while True:
        if keyboard.is_pressed('q'):
            mask = np.zeros(frame_shape, dtype='uint8')
            img_pil = Image.fromarray(mask)
            dra = ImageDraw.Draw(img_pil)
            dra.text((1000, 160), prac_char, font=font, fill=(1, 1, 1))
            mask = np.array(img_pil)

        success1, frame1 = cap.read()
        success2, frame2 = video_capture2.read()

        if not success1:
            break

        if not success2:
            video_capture2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame2 = video_capture2.read()

        frame1 = cv2.flip(frame1, 1)
        rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        op = hand_landmark.process(rgb)

        if op.multi_hand_landmarks:
            for all_landmarks in op.multi_hand_landmarks:
                draw.draw_landmarks(frame1, all_landmarks, hands.HAND_CONNECTIONS)

                x1 = int(all_landmarks.landmark[8].x * frame_shape[1])
                y1 = int(all_landmarks.landmark[8].y * frame_shape[0])
                x2 = int(all_landmarks.landmark[4].x * frame_shape[1])
                y2 = int(all_landmarks.landmark[4].y * frame_shape[0])

                d1 = math.dist([x1, y1], [x2, y2])

                if d1 < 55 and prevxy is not None:
                    if math.dist(prevxy, [x1, y1]) > 5:
                        if 1000 < x1 < 1720 and 150 < y1 < 900:
                            cv2.line(mask, prevxy, (x1, y1), colour, thickness)
                prevxy = (x1, y1)

        frame1 = cv2.resize(frame1, (1920, 1080))
        frame1 = np.where(mask, mask, frame1)
        frame1 = cv2.resize(frame1, (1050, 700))
        frame2 = cv2.resize(frame2, (450, 700))

        side_by_side = cv2.hconcat([frame2, frame1])

        ret, jpeg = cv2.imencode('.jpg', side_by_side)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def eva():
    global cap
    global ra
    global imgCanvas

    cap = cv2.VideoCapture(0)

    eva_mask = np.zeros(frame_shape, dtype='uint8')
    imgCanvas = np.zeros((1080, 1920, 3), dtype='uint8')
    cv2.rectangle(eva_mask, (900, 150), (1620, 900), (1, 1, 1), 2)

    img_path = f'.\\static\\char\\{char}_img.jpg'

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(max_num_hands=1)

    draw = mp.solutions.drawing_utils

    colour = (1, 1, 1)
    thickness = 63

    prevxy = None

    while True:
        if keyboard.is_pressed('q'):
            eva_mask = np.zeros(frame_shape, dtype='uint8')
            cv2.rectangle(eva_mask, (900, 150), (1620, 900), (1, 1, 1), 2)

        success1, frame1 = cap.read()

        if not success1:
            break

        frame1 = cv2.flip(frame1, 1)
        rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        op = hand_landmark.process(rgb)

        if op.multi_hand_landmarks:
            for all_landmarks in op.multi_hand_landmarks:
                draw.draw_landmarks(frame1, all_landmarks, hands.HAND_CONNECTIONS)

                x1 = int(all_landmarks.landmark[8].x * frame_shape[1])
                y1 = int(all_landmarks.landmark[8].y * frame_shape[0])
                x2 = int(all_landmarks.landmark[4].x * frame_shape[1])
                y2 = int(all_landmarks.landmark[4].y * frame_shape[0])

                d1 = math.dist([x1, y1], [x2, y2])

                if d1 < 55 and prevxy is not None:
                    if math.dist(prevxy, [x1, y1]) > 5:
                        if 1620 > x1 > 900 > y1 > 150:
                            cv2.line(eva_mask, prevxy, (x1, y1), colour, thickness)
                            cv2.line(imgCanvas, prevxy, (x1, y1), (255, 255, 255), thickness)
                prevxy = (x1, y1)

        if ra == 1:
            frame1 = cv2.resize(frame1, (1920, 1080))
            frame1 = np.where(eva_mask, eva_mask, frame1)
            frame1 = cv2.resize(frame1, (1050, 700))
            frame2 = cv2.imread(img_path)
            frame2 = cv2.resize(frame2, (450, 700))

            side_by_side = cv2.hconcat([frame2, frame1])

            ret, jpeg = cv2.imencode('.jpg', side_by_side)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        else:
            frame1 = cv2.resize(frame1, (1920, 1080))
            frame1 = np.where(eva_mask, eva_mask, frame1)

            ret, jpeg = cv2.imencode('.jpg', frame1)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def lrn():
    global cap, lrn_toggle

    lrn_char = chr(random.randint(ord('A'), ord('Z')))

    cap = cv2.VideoCapture(0)
    video_capture2 = cv2.VideoCapture(f'.\\static\\char\\{lrn_char}.mp4')
    img_path = f'.\\static\\char\\{lrn_char}_img.jpg'

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(max_num_hands=1)

    draw = mp.solutions.drawing_utils

    colour = (1, 1, 1)
    thickness = 63
    font_size = 900
    font = ImageFont.truetype("./static/font/font.otf", font_size)

    lrn_mask = np.zeros(frame_shape, dtype='uint8')
    img_pil = Image.fromarray(lrn_mask)
    dra = ImageDraw.Draw(img_pil)
    dra.text((1000, 160), lrn_char, font=font, fill=(1, 1, 1))
    lrn_mask = np.array(img_pil)

    prevxy = None

    while True:
        if keyboard.is_pressed('q'):
            lrn_mask = np.zeros(frame_shape, dtype='uint8')
            img_pil = Image.fromarray(lrn_mask)
            dra = ImageDraw.Draw(img_pil)
            dra.text((1000, 160), lrn_char, font=font, fill=(1, 1, 1))
            lrn_mask = np.array(img_pil)

        success1, frame1 = cap.read()
        if not lrn_toggle:
            success2, frame2 = video_capture2.read()

            if not success2:
                video_capture2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, frame2 = video_capture2.read()

        if not success1:
            break

        frame1 = cv2.flip(frame1, 1)
        rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        op = hand_landmark.process(rgb)

        if op.multi_hand_landmarks:
            for all_landmarks in op.multi_hand_landmarks:
                draw.draw_landmarks(frame1, all_landmarks, hands.HAND_CONNECTIONS)

                x1 = int(all_landmarks.landmark[8].x * frame_shape[1])
                y1 = int(all_landmarks.landmark[8].y * frame_shape[0])
                x2 = int(all_landmarks.landmark[4].x * frame_shape[1])
                y2 = int(all_landmarks.landmark[4].y * frame_shape[0])

                d1 = math.dist([x1, y1], [x2, y2])

                if d1 < 55 and prevxy is not None:
                    if math.dist(prevxy, [x1, y1]) > 5:
                        if 1000 < x1 < 1720 and 150 < y1 < 900:
                            cv2.line(lrn_mask, prevxy, (x1, y1), colour, thickness)
                prevxy = (x1, y1)

        frame1 = cv2.resize(frame1, (1920, 1080))
        frame1 = np.where(lrn_mask, lrn_mask, frame1)
        frame1 = cv2.resize(frame1, (1050, 700))
        if lrn_toggle:
            frame2 = cv2.imread(img_path)
        frame2 = cv2.resize(frame2, (450, 700))

        side_by_side = cv2.hconcat([frame2, frame1])

        ret, jpeg = cv2.imencode('.jpg', side_by_side)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/learn')
def learn():
    global cap
    cap.release()
    return render_template('learn.html')


@app.route('/learn/start')
def start_learn():
    return render_template('start_learn.html')


@app.route('/lrn_feed')
def lrn_feed():
    return Response(lrn(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/lrn_toggle', methods=['POST'])
def lrn_toggle():
    global lrn_toggle
    lrn_toggle = not lrn_toggle
    return 'OK'


@app.route('/evaluate')
def evaluate():
    session['page_load_count'] = 0
    global ch, accuracy, k, char, ra, cha, raa, cap
    k = 0
    accuracy = 0
    ch = {}
    # c = ['A', 'B', 'C', 'D']
    cha = ['C', 'B', 'D']
    raa = [1, 0, 1]
    char = cha[k]
    ra = raa[k]
    cap.release()
    return render_template('evaluate.html')


def fun():
    global imgCanvas, accuracy, ch, k, char, ra
    ima = cv2.resize(imgCanvas, (28, 28))
    ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    ima = ima.reshape((28, 28, 1))
    ima = ima.astype('float32') / 255
    val = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(np.array([ima])))])

    kx = k - 1
    accuracy += 1
    ch[cha[kx]] = val

    k += 1
    if k < 3:
        char = cha[k]
        ra = raa[k]
    return


@app.route('/evaluate/start_next')
def start_eva():
    fun()
    return start_evaluate()


@app.route('/evaluate/start')
def start_evaluate():
    global ch, accuracy
    if 'page_load_count' not in session:
        session['page_load_count'] = 0

    session['page_load_count'] += 1

    if session['page_load_count'] <= 3:
        return render_template('start_evaluate.html')
    else:
        print(ch)
        ch = {}
        accuracy = 0
        session.pop('page_load_count', None)
        return redirect(url_for('result_evaluate'))


@app.route('/evaluate/result')
def result_evaluate():
    global cap
    cap.release()
    return render_template('result_evaluate.html')


@app.route('/eval_feed')
def eval_feed():
    return Response(eva(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/practice')
def practice():
    global cap
    cap.release()
    return render_template('practice.html')


@app.route('/practice/start')
def start_practice():
    return render_template('start_practice.html')


@app.route('/prac_feed')
def prac_feed():
    return Response(prac(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/get_char")
def get_char():
    global char
    return jsonify({"variable": char})


@app.route("/get_ra")
def get_ra():
    global ra
    return jsonify({"variable": ra})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
