import RPi.GPIO as GPIO
import cv2 as cv
import numpy as np



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
    wheels[0].ChangeDutyCycle(60)
    wheels[1].ChangeDutyCycle(60)

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
    wheels[2].ChangeDutyCycle(15)
    wheels[3].ChangeDutyCycle(15)

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


def main():
    camera = cv.VideoCapture(0)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 400)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 300)
    path = "/home/pi/Study/project/camera/data4/data4"
    i = 0
    carState = "stop"

    while (camera.isOpened()):
        # delay = int(1000/camera.get(cv.CAP_PROP_FPS))
        keyValue = cv.waitKey(1)

        if keyValue & 0xFF == 27:
            print('finish')
            break

        elif keyValue == 119:
            print("go")
            carState = "go"
            wheel_go()

        elif keyValue == 115:
            print("back")
            carState = "back"
            wheel_back()

        elif keyValue == 100:
            print("right")
            carState = "right"
            wheel_right()

        elif keyValue == 97:
            print("left")
            carState = "left"
            wheel_left()

        elif keyValue == 32:
            print("stop")
            carState = "stop"
            wheel_stop()
        
        _, img = camera.read()
        #cv.imshow("Original", img)

        height, _, _ = img.shape
        img_cut = img[int(height/2):, :, :]

        gray_img = cv.cvtColor(img_cut, cv.COLOR_BGR2GRAY)

        blur_img = cv.GaussianBlur(gray_img, (7,7), 0)

        _, binary_img = cv.threshold(blur_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        canny_img = cv.Canny(blur_img, 100, 150)



        cv.imshow("Original", img)

        if carState == "left":
            cv.imwrite("%s_%05d_%03d.png" %(path, i, 45), canny_img)
            i += 1 

        elif carState == "right":
            cv.imwrite("%s_%05d_%03d.png" %(path, i, 135), canny_img)
            i += 1

        elif carState == "go":
            cv.imwrite("%s_%05d_%03d.png" %(path, i, 90), canny_img)
            i += 1
    
    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    GPIO.cleanup()
