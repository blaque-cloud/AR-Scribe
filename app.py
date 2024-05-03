import math
import time
import random
import cv2
import keyboard
import mediapipe as mp
import numpy as np
import pandas as pd
import ast
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, redirect, url_for, session, jsonify, request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'secret_key'

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
lrn_toggle = True
prac_toggle = False
accuracy = 0
name = ''
result = []
k = 0
prac_time = -1
char = ''
lrn_char = ''
ra = 0
cha = []
raa = []
pred = []
lrn_ch = -1
frame_shape = (1080, 1920, 3)
imgCanvas = np.zeros((1080, 1920, 3), dtype='uint8')
AlphaMODEL = load_model("./static/models/alpha.h5")
AlphaLABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ''}

t = 0

time_list = []

def prac():
    global cap, prac_toggle

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

    prac_mask = np.zeros(frame_shape, dtype='uint8')
    img_pil = Image.fromarray(prac_mask)
    dra = ImageDraw.Draw(img_pil)
    dra.text((150, 160), prac_char, font=font, fill=(1, 1, 1))
    prac_mask = np.array(img_pil)
    cv2.rectangle(prac_mask, (100, 150), (820, 900), (1, 1, 1), 2)
    cv2.rectangle(prac_mask, (1000, 150), (1720, 900), (1, 1, 1), 2)

    prevxy = None

    while True:
        if keyboard.is_pressed('q'):
            prac_mask = np.zeros(frame_shape, dtype='uint8')
            img_pil = Image.fromarray(prac_mask)
            dra = ImageDraw.Draw(img_pil)
            dra.text((150, 160), prac_char, font=font, fill=(1, 1, 1))
            prac_mask = np.array(img_pil)
            cv2.rectangle(prac_mask, (100, 150), (820, 900), (1, 1, 1), 2)
            cv2.rectangle(prac_mask, (1000, 150), (1720, 900), (1, 1, 1), 2)

        success1, frame1 = cap.read()

        if not success1:
            break

        if prac_toggle:
            success2, frame2 = video_capture2.read()            
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
                        if x1 > 100 and x1 < 820 and y1 > 150 and y1 < 900:
                            cv2.line(prac_mask, prevxy, (x1, y1), colour, thickness)
                        if x1 > 1000 and x1 < 1720 and y1 > 150 and y1 < 900:
                            cv2.line(prac_mask, prevxy, (x1, y1), colour, thickness)
                prevxy = (x1, y1)

        frame1 = cv2.resize(frame1, (1920, 1080))
        frame1 = np.where(prac_mask, prac_mask, frame1)

        if prac_toggle:
            frame1 = cv2.resize(frame1, (1450, 700))
            frame2 = cv2.resize(frame2, (450, 700))

            side_by_side = cv2.hconcat([frame2, frame1])
        else:
            side_by_side = cv2.hconcat([frame1])

        ret, jpeg = cv2.imencode('.jpg', side_by_side)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def eva():
    global cap
    global ra
    global imgCanvas
    global t
    cap = cv2.VideoCapture(0)

    eva_mask = np.zeros(frame_shape, dtype='uint8')
    imgCanvas = np.zeros((1080, 1920, 3), dtype='uint8')
    cv2.rectangle(eva_mask, (900, 150), (1620, 900), (1, 1, 1), 2)

    if ra == 2:
        img_path = f'.\\static\\char\\{char}_img.jpg'
    elif ra == 0:
        img_path = f'.\\static\\char\\eval\\{char}.jpg'

    hands = mp.solutions.hands
    hand_landmark = hands.Hands(max_num_hands=1)

    draw = mp.solutions.drawing_utils

    colour = (1, 1, 1)
    thickness = 63

    prevxy = None

    while True:
        if keyboard.is_pressed('q'):
            eva_mask = np.zeros(frame_shape, dtype='uint8')
            imgCanvas = np.zeros((1080, 1920, 3), dtype='uint8')
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

            ret, jpeg = cv2.imencode('.jpg', frame1)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
        else:
            frame1 = cv2.resize(frame1, (1920, 1080))
            frame1 = np.where(eva_mask, eva_mask, frame1)
            frame1 = cv2.resize(frame1, (1050, 700))
            frame2 = cv2.imread(img_path)
            frame2 = cv2.resize(frame2, (450, 700))

            side_by_side = cv2.hconcat([frame2, frame1])

            ret, jpeg = cv2.imencode('.jpg', side_by_side)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def lrn():
    global cap, lrn_toggle, lrn_ch, lrn_char

    lrn_toggle = True
    lrn_ch += 1
    lrn_char = chr(ord('A') + lrn_ch)

    cap = cv2.VideoCapture(0)
    video_capture2 = cv2.VideoCapture(f'.\\static\\char\\{lrn_char}.mp4')
    img_path = f'.\\static\\char\\learn\\{lrn_char}.jpg'

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

    start_time = time.time()

    while True:

        if not lrn_toggle:
            if time.time() - start_time > 40:
                lrn_toggle = not lrn_toggle
                start_time = time.time()
        else:
            if time.time() - start_time > 5:
                lrn_toggle = not lrn_toggle
                start_time = time.time()

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
    global lrn_ch
    if lrn_ch < 5:
        return render_template('start_learn.html')
    else:
        return render_template('index.html')


@app.route('/lrn_feed')
def lrn_feed():
    return Response(lrn(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/lrn_toggle', methods=['POST'])
def lrn_toggle():
    global lrn_toggle
    lrn_toggle = not lrn_toggle
    return 'OK'


@app.route("/get_lrnchar")
def get_lrnchar():
    global lrn_char
    return jsonify({"variable": lrn_char})


@app.route('/evaluate')
def evaluate():
    global result, accuracy, k, char, ra, cha, raa, cap, imgCanvas

    ima = cv2.resize(imgCanvas, (28, 28))
    ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    ima = ima.reshape((28, 28, 1))
    ima = ima.astype('float32') / 255
    val = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(np.array([ima])))])

    session['page_load_count'] = 0
    k = 0
    accuracy = 0
    result = []
    c = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    cha = random.choices(c, k = 12)
    raa = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    char = cha[k]
    ra = raa[k]
    cap.release()
    return render_template('evaluate.html')


def fun():
    global imgCanvas, accuracy, result, k, char, ra, cha, raa, pred

    mask = imgCanvas[150:900, 900:1620]
    mask = cv2.resize(mask, (28, 28))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.reshape((28,28,1))
    mask = mask.astype('float32') / 255

    val = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(mask.reshape(1,28,28,1)))])

    pred.append(val)

    accuracy += 1
    if val == char:
        result.append(1)
    else:
        result.append(0)

    k += 1
    if k < 12:
        char = cha[k]
        ra = raa[k]
    return


@app.route('/evaluate/start_next')
def start_eva():
    fun()
    return start_evaluate()


@app.route('/evaluate/start')
def start_evaluate():
    global result, t, pred, time_list
    if 'page_load_count' not in session:
        session['page_load_count'] = 0

    t = time.time()
    session['page_load_count'] += 1

    if session['page_load_count'] <= 12:
        return render_template('start_evaluate.html')
    else:
        print(cha)
        print(pred)
        print(result)
        print(time_list)
        session.pop('page_load_count', None)

        with open('eva_iter.txt', 'r') as file:
            c=file.read()
        file.close()
        c=ast.literal_eval(c)

        if name in c:
            c[name]+=1
        else:
            c[name]=1

        temp1 = c[name]

        with open('prac_iter.txt', 'r') as file:
            d=file.read()
        file.close()
        d=ast.literal_eval(d)

        if name not in d:
            d[name]=0

        temp2 = d[name]
        with open('eva_iter.txt', 'w') as file:
            file.write(str(c))

        if not os.path.isfile(f'result/eval/eval_{name}.xlsx'):
            df = pd.DataFrame({})

            df.to_excel(f'result/eval/eval_{name}.xlsx', index=False)

        df = pd.read_excel(f'result/eval/eval_{name}.xlsx')

        new_data = pd.DataFrame({
            'Iteration': [temp1]*12,
            'Times Practiced': [temp2]*12,
            'Time Taken': time_list,
            'Result': result
            })
        
        df = df.append(new_data, ignore_index=True)
        
        df.to_excel(f'result/eval/eval_{name}.xlsx', index=False)
        
        pred = []
        time_list = []

        return redirect(url_for('result_evaluate'))


@app.route('/get_name', methods=['POST'])
def get_name():
    global name
    name = request.form.get('name')
    return start_evaluate()


@app.route('/tim', methods=['POST'])
def tim():
    global t
    time_list.append(round((time.time() - t - 2), 2))
    return 'OK'


@app.route('/evaluate/result')
def result_evaluate():
    global cap
    cap.release()
    return render_template('result_evaluate.html')


@app.route("/get_result")
def get_result():
    global result
    return jsonify({"l1": sum(result[0:4]), "l2": sum(result[4:8]), "l3": sum(result[8:12])})


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
    global prac_time

    with open('prac_iter.txt', 'r') as file:
        c=file.read()
    file.close()

    c=ast.literal_eval(c)
    if name in c:
        c[name]+=1
    else:
        c[name]=1
    temp = c[name]

    with open('prac_iter.txt', 'w') as file:
        file.write(str(c))
    file.close()

    if not os.path.isfile(f'result/prac/prac_{name}.xlsx'):
        df = pd.DataFrame({})

        df.to_excel(f'result/prac/prac_{name}.xlsx', index=False)

    df = pd.read_excel(f'result/prac/prac_{name}.xlsx')

    if temp != 1:
        tt = receive_tt()
        tt = round(prac_time - time.time(), 2)

        new_data = pd.DataFrame({
            'Iteration': temp-1,
            'Time Taken': tt,
            'Result': random.choices([0, 1])
            })
        
        df = df.append(new_data, ignore_index=True)

    df.to_excel(f'result/prac/prac_{name}.xlsx', index=False)
    
    prac_time = time.time()
    return render_template('start_practice.html')


@app.route('/receive_tt', methods=['POST'])
def receive_tt():
    tt = request.form.get('tt')
    return tt


@app.route('/prac_feed')
def prac_feed():
    return Response(prac(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prac_toggle', methods=['POST'])
def prac_toggle():
    global prac_toggle
    prac_toggle = not prac_toggle
    return 'OK'


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
