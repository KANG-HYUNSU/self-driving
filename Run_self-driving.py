import RPi.GPIO as GPIO
import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
import tensorflow as tf

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

CCW_L = [12,20,18,21]
CCW_R = [13,22,19,23]
wheels = []
MOT_PREQ = 100

GPIO.setup(CCW_L, GPIO.OUT)
GPIO.setup(CCW_R, GPIO.OUT)
GPIO.output(CCW_L, False)
GPIO.output(CCW_R, False)

wheel_L = GPIO.PWM(CCW_L[1], MOT_PREQ)
wheel_L.start(0.0)
wheels.append(wheel_L)  
wheel_L_B = GPIO.PWM(CCW_L[3], MOT_PREQ)
wheel_L_B.start(0.0)
wheels.append(wheel_L_B)

wheel_R = GPIO.PWM(CCW_R[1], MOT_PREQ)
wheel_R.start(0.0)
wheels.append(wheel_R)
wheel_R_B = GPIO.PWM(CCW_R[3], MOT_PREQ)
wheel_R_B.start(0.0)
wheels.append(wheel_R_B)

def wheel_go():
    GPIO.output(CCW_L[0], True)
    GPIO.output(CCW_L[2], True)
    wheels[0].ChangeDutyCycle(70)
    wheels[1].ChangeDutyCycle(70)

    GPIO.output(CCW_R[0], True)
    GPIO.output(CCW_R[2], True)
    wheels[2].ChangeDutyCycle(70)
    wheels[3].ChangeDutyCycle(70)

def wheel_back():
    GPIO.output(CCW_L[0], False)
    GPIO.output(CCW_L[2], False)
    wheels[0].ChangeDutyCycle(20)
    wheels[1].ChangeDutyCycle(20)

    GPIO.output(CCW_R[0], False)
    GPIO.output(CCW_R[2], False)
    wheels[2].ChangeDutyCycle(20)
    wheels[3].ChangeDutyCycle(20)

def wheel_right():
    GPIO.output(CCW_L[0], True)
    GPIO.output(CCW_L[2], True)
    wheels[0].ChangeDutyCycle(15)
    wheels[1].ChangeDutyCycle(15)

    GPIO.output(CCW_R[0], True)
    GPIO.output(CCW_R[2], True)
    wheels[2].ChangeDutyCycle(95)
    wheels[3].ChangeDutyCycle(95)

def wheel_left():
    GPIO.output(CCW_L[0], True)
    GPIO.output(CCW_L[2], True)
    wheels[0].ChangeDutyCycle(95)
    wheels[1].ChangeDutyCycle(95)

    GPIO.output(CCW_R[0], True)
    GPIO.output(CCW_R[2], True)
    wheels[2].ChangeDutyCycle(15)
    wheels[3].ChangeDutyCycle(15)

def wheel_stop():
    GPIO.output(CCW_L[0], False)
    GPIO.output(CCW_L[2], False)
    wheels[0].ChangeDutyCycle(0)
    wheels[1].ChangeDutyCycle(0)

    GPIO.output(CCW_R[0], False)
    GPIO.output(CCW_R[2], False)
    wheels[2].ChangeDutyCycle(0)
    wheels[3].ChangeDutyCycle(0)


# TensorFlow Lite 모델 로드
interpreter = tflite.Interpreter(model_path='/home/pi/Study/project/model_4_2.tflite')
interpreter.allocate_tensors()

# 입력 및 출력 텐서 인덱스 얻기
input_tensor_index = interpreter.get_input_details()[0]['index']
output_tensor_index = interpreter.get_output_details()[0]['index']

input_tensor_details = interpreter.get_input_details()[0]
print("Input Tensor Shape:", input_tensor_details['shape'])

def main():
    try:
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 400)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 300)

        while (cap.isOpened()):
            delay = int(1000/cap.get(cv.CAP_PROP_FPS))
            ret, frame = cap.read()

            if not ret:
                print('No frame')
                break

            cv.imshow("Driveing", frame)

            # 이미지 전처리 및 추론
            height, _, _ = frame.shape
            frame_cut = frame[int(height/2):, :, :]

            gray_frame = cv.cvtColor(frame_cut, cv.COLOR_BGR2GRAY)

            blur_frame = cv.GaussianBlur(gray_frame, (7,7), 0)

            _, binary_frame = cv.threshold(blur_frame, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

            canny_frame = cv.Canny(binary_frame, 100, 150)

            frame = canny_frame / 255.0
            frame = np.expand_dims(frame, axis=0)   # Add a batch dimension
            frame = np.expand_dims(frame, axis=-1)  # Add the channel dimension
            frame = frame.astype(np.float32)

            interpreter.set_tensor(input_tensor_index, frame)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_tensor_index)

            predicted_angle = output_data[0][0]
            print(f'예측 각도: {predicted_angle}')

            if 68 < predicted_angle < 128:
                print('go')
                wheel_go()

            elif 131 < predicted_angle < 145:
                print("right")
                wheel_right()

            elif 40 < predicted_angle < 55:
                print('left')
                wheel_left()

            else:
                print('stop')
                wheel_stop()

            
            keyValue = cv.waitKey(delay)

            if keyValue & 0xFF == 27:
                print('finish')
                break

    except Exception as e:
        print(f"An error occurred: {e}")


    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
    GPIO.cleanup()