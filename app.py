import math
import random
import cv2
import keyboard
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, redirect, url_for, session, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

accuracy = 0
ch = {}


def prac(flag):
    cap = cv2.VideoCapture(0)
    if flag == 1:
        cap.release()
    # frame_shape = (1080, 1920, 3)
    mask = np.zeros(frame_shape, dtype='uint8')
    imgCanvas = np.zeros((1080, 1920, 3), dtype='uint8')

    prac_char = chr(random.randint(ord('A'), ord('Z')))
    video_capture2 = cv2.VideoCapture(f'.\\static\\char\\{prac_char}.mp4')

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(max_num_hands=1)

    draw = mp.solutions.drawing_utils

    colour = (1, 1, 1)
    thickness = 63

    font_size = 900
    font = ImageFont.truetype("./static/font/font.otf", font_size)

    prevxy = None

    AlphaMODEL = load_model("./static/models/alpha.h5")
    AlphaLABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                   10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                   20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ''}

    while True:
        if keyboard.is_pressed('q'):
            mask = np.zeros(frame_shape, dtype='uint8')

        mask = cv2.resize(mask, (1920, 1080))
        cap.set(3, 1920)
        cap.set(4, 1080)
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
                            cv2.line(imgCanvas, prevxy, (x1, y1), (255, 255, 255), thickness)
                prevxy = (x1, y1)

        frame1 = cv2.resize(frame1, (1920, 1080))
        frame1 = np.where(mask, mask, frame1)
        frame1 = cv2.resize(frame1, (1050, 700))
        frame2 = cv2.resize(frame2, (450, 700))

        img_pil = Image.fromarray(mask)
        dra = ImageDraw.Draw(img_pil)
        dra.text((1000, 160), prac_char, font=font, fill=(1, 1, 1))
        mask = np.array(img_pil)
        mask = cv2.resize(mask, (922, 700))

        if keyboard.is_pressed("a"):
            ima = cv2.resize(imgCanvas, (28, 28))
            ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            ima = ima.reshape((28, 28, 1))
            ima = ima.astype('float32') / 255
            val = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(np.array([ima])))])
            print(val)

        side_by_side = cv2.hconcat([frame2, frame1])

        ret, jpeg = cv2.imencode('.jpg', side_by_side)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


frame_shape = (1080, 1920, 3)
eva_mask = np.zeros(frame_shape, dtype='uint8')
nex = 0
rand = 0

# char = chr(random.randint(ord('A'), ord('Z')))
c = ['A', 'B', 'C', 'D']
char = "A"


def eva(flag):
    global accuracy, char
    global ch
    global nex
    global rand

    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    if flag == 1:
        cap.release()
    global eva_mask
    eva_mask = np.zeros(frame_shape, dtype='uint8')
    imgCanvas = np.zeros((1080, 1920, 3), dtype='uint8')
    cv2.rectangle(eva_mask, (900, 150), (1620, 900), (1, 1, 1), 2)

    if rand == 0:
        char = c[random.randint(0, 3)]

    ra = random.randint(0, 1)
    ra = 0
    if ra == 1:
        img_path = f'.\\static\\char\\{char}_img.jpg'
    else:
        img_path = f'.\\static\\sounds\\A.mp3'

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(max_num_hands=1)

    draw = mp.solutions.drawing_utils

    colour = (1, 1, 1)
    thickness = 63

    prevxy = None

    AlphaMODEL = load_model("./static/models/alpha.h5")
    AlphaLABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                   10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                   20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ''}

    while True:
        if keyboard.is_pressed('q'):
            eva_mask = np.zeros(frame_shape, dtype='uint8')
            cv2.rectangle(eva_mask, (900, 150), (1620, 900), (1, 1, 1), 2)

        eva_mask = cv2.resize(eva_mask, (1920, 1080))
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

        if nex == 1:
            rand = 0
            nex = 0
            cv2.imshow("", imgCanvas)
            ima = cv2.resize(imgCanvas, (28, 28))
            ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            ima = ima.reshape((28, 28, 1))
            ima = ima.astype('float32') / 255
            val = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(np.array([ima])))])

            accuracy += 1
            ch[char] = val

            print(accuracy)
            print(ch)
            if accuracy == 3:
                accuracy = 0
                ch = {}

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


def lrn(flag):
    return


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/learn')
def learn():
    lrn(1)
    return render_template('learn.html')


@app.route('/learn/start')
def start_learn():
    return render_template('start_learn.html')


@app.route('/evaluate')
def evaluate():
    session['page_load_count'] = 0
    global ch, accuracy
    accuracy = 0
    ch = {}
    eva(1)
    return render_template('evaluate.html')


@app.route('/evaluate/start_next')
def start_eva():
    global nex
    nex = 1
    return start_evaluate(1)


@app.route('/evaluate/start')
def start_evaluate(fla=1):
    global char
    if 'page_load_count' not in session:
        session['page_load_count'] = 0

    if fla == 1:
        session['page_load_count'] += 1

    if session['page_load_count'] <= 3:
        return render_template('start_evaluate.html', char=char)
    else:
        session.pop('page_load_count', None)
        return redirect(url_for('result_evaluate'))


@app.route('/clear_feed')
def clear_feed():
    global eva_mask, rand
    rand = 1
    eva_mask = np.zeros(frame_shape, dtype='uint8')
    cv2.rectangle(eva_mask, (900, 150), (1620, 900), (1, 1, 1), 2)
    return start_evaluate(0)


@app.route('/evaluate/result')
def result_evaluate():
    return render_template('result_evaluate.html')


@app.route('/eval_feed')
def eval_feed():
    return Response(eva(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/practice')
def practice():
    prac(1)
    return render_template('practice.html')


@app.route('/practice/start')
def start_practice():
    return render_template('start_practice.html')


@app.route('/prac_feed')
def prac_feed():
    return Response(prac(0), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/get_char")
def get_char():
    global char
    return jsonify({"variable": char})


if __name__ == '__main__':
    app.run(debug=True)
