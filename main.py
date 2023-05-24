import os
import sys
import cv2
import time
import glob
import keyboard
import pyautogui
import numpy as np
from PIL import Image
from numpy import loadtxt
from keras.models import Model 
from keras import backend as K
from keras.models import load_model
from keras.layers import Activation
from screeninfo import get_monitors
from pynput.mouse import Button, Controller
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from keras.layers import BatchNormalization, Dropout, Dense

mouse = Controller()

model = EfficientNetB0(include_top=False, input_shape=(150, 150, 3), pooling='max', weights='imagenet')
x = model.output
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(384)(x)
x = BatchNormalization()(x)
x = Activation('swish')(x)
x = Dropout(0.5)(x)
x = Dense(192)(x)
x = Activation('swish')(x)
x = Dense(96)(x)
x = Activation('swish')(x)
x = Dense(48)(x)
x = Activation('swish')(x)
x = Dense(24)(x)
x = Activation('swish')(x)
x = Dense(12)(x)
x = Activation('swish')(x)
x = Dense(6)(x)
x = Activation('swish')(x)
predictions = Dense(3, activation='softmax')(x)
model_final = Model(inputs = model.input, outputs = predictions)
model_final.load_weights('Final_Model_Weight.h5')

camera = cv2.VideoCapture(0)
def capture_image():
    return_value,frame = camera.read()

    return frame

def resolution_scaling(x,y):
    monitors = get_monitors()
    new_x=monitors[0].width             #input("Enter Current Monitor Horizontal Resoultion: ")
    new_y=monitors[0].height              #input("Enter Current Monitor Vertical Resoultion: ")
    
    var_x=int((x/1280)*new_x)
    var_y=int((y/720)*new_y)
    
    return var_x,var_y

#0=left Click
#1=Hover
#2=Right Click
import cv2
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


_, frame = cap.read()

h, w, c = frame.shape

#print(h,w)

while True:
    if keyboard.is_pressed('q'):  # Change 'q' to the desired key
        break
    _, frame = cap.read()
    frame= cv2.flip(frame,1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(frame, (x_min-15, y_min-15), (x_max+15, y_max+15), (0, 255, 0), 2)  
            label= cv2.rectangle(frame, (x_min-15, y_min-15), (x_max+15, y_max+15), (0, 255, 0), 2)
            #cv2.rectangle(frame, (70, 70), (w-70,h-70), (0, 0,255), 2)
            
            start =time.time()
            
            img_crop=frame[y_min-15:y_max+15,x_min-15:x_max+15]
            
            try:
                img_crop = cv2.cvtColor(img_crop,cv2.COLOR_BGR2RGB)
                
            except:
                #print("Next entry.")
                continue

                
            cv2.imwrite('img_crop/' + 'crp' + '.jpg', img_crop)
            address = r'img_crop\crp.jpg'
            Images = glob.glob(address)
            
            img_crop = Image.open(Images[0]).convert("RGB")
            img_crop = img_crop.resize((150, 150))
            img_crop = np.array(img_crop)
            img_crop = np.array(img_crop, dtype='float32')
            img_crop = img_crop/255
            img_crop = img_crop.reshape((1, 150, 150, 3))
            
            test_datagen = ImageDataGenerator()
            test_generator = test_datagen.flow(img_crop, shuffle=False)
            
            predict_test = model_final.predict(test_generator)
            predicted_label = np.argmax(predict_test)
            
            end= time.time()
            
            if predicted_label==0:
                p_l="Left Click"
                mouse.click(Button.left)
                mouse.release(Button.left)
                
            elif predicted_label==1:
                p_l="Hover"
                
                a,b=resolution_scaling(x,y)
                
                try:
                    pyautogui.moveTo(a,b,.08,pyautogui.easeInQuad)
                except:
                    continue
                                                        
            elif predicted_label==2:
                p_l="Right Click"
                mouse.press(Button.right)    
                mouse.release(Button.right)
            
            cv2.putText(label, str(p_l), (x-60, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            #label= cv2.flip(label,1)
            #print(predicted_label)
            
            fps=1/(end-start)
            
            #string formatting for fps
            cv2.putText(frame,f"{fps:.2f} FPS",(20,30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)  
            
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()        
cap.release()