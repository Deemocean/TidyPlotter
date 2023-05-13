import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import numpy as np
import time
from threading import Thread
import cv2
import math






offset = 3.2625

max_height = 12 #maximum hieght drawing in cm
max_width = 12 #maximum width of drawing in cm

img = cv2.imread('imgs/face.png', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img,100,200)
height, width = edges.shape
height = float(height)
width = float(width)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

normalizing_factor = 0.0
scaling_factor = 0.0

if height >= width:
    normalizing_factor = height
    scaling_factor = max_height
else:
    normalizing_factor = width
    scaling_factor = max_width

#converts the points stored in contours from ints to floats
contours_float = []
for contour in contours:
    contour_float = contour.astype(np.float32)
    contours_float.append(contour_float)

#normalizes all points with respect to width and height of the image
for contour in contours_float:
    contour[:, :, 0] /= normalizing_factor
    contour[:, :, 1] /= normalizing_factor

#rescales points to fit page size (10 was just used for debugging)
for contour in contours_float:
    contour[:, :, 0] *= scaling_factor
    contour[:, :, 1] *= scaling_factor

# Get x and y coordinates of contour points
x_coords = []
y_coords = []
for contour in contours_float:
    for point in contour:
        x = point[0][0]
        y = point[0][1]
        x_coords.append(x)
        y_coords.append(y)

coordinates = [[x, y] for x, y in zip(x_coords, y_coords)]

#print(coordinates)

l1 = 11.5 #length of first arm
l2 = 13 #length of second arm

#define GPIO pins
GPIO_pins = (-1, -1, -1) # Microstep Resolution MS1-MS3 -> GPIO Pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)



lift=GPIO.PWM(18, 50)
lift.start(0)

direction1= 21       # Direction -> GPIO Pin
step1= 20       # Direction -> GPIO Pin

step2 = 24      # Step -> GPIO Pin
direction2= 23       # Direction -> GPIO Pin




class Arm:
  def __init__(self,angle, motor):
    self.angle = angle
    self.motor = motor
  
  def to_angle(self, angle):
    angle_diff = np.abs(angle-self.angle)
    # step = int(angle_diff*200/360)

    step = int(angle_diff*200/360)*4

    if(self.angle>=angle):
      self.motor.motor_go(True, "1/4" , step, 0.01, False, .05)

    else :
      self.motor.motor_go(False, "1/4" , step, 0.01,False, .05)

    self.angle=angle


# def anglecalc(x, y):

#   link1_length =l1
#   link2_length = l2
#   # Find the distance from the origin to the target point
#   distance = math.sqrt(x**2 + y**2)

#   # Find the angle between the x-axis and the line connecting the origin to the target point
#   theta1 = math.atan2(y, x)

#   # Find the angle between the two links using the law of cosines
#   cos_theta2 = (link1_length**2 + link2_length**2 - distance**2) / (2 * link1_length * link2_length)
#   sin_theta2 = math.sqrt(1 - cos_theta2**2)
#   theta2 = math.atan2(sin_theta2, cos_theta2)

#   # Find the angle between the x-axis and the first link using the law of cosines
#   cos_theta3 = (link1_length**2 + distance**2 - link2_length**2) / (2 * link1_length * distance)
#   sin_theta3 = math.sqrt(1 - cos_theta3**2)
#   theta3 = math.atan2(sin_theta3, cos_theta3)

#   print("angle1: "+str(math.degrees(theta1)))
#   print("angle2: "+str(180-math.degrees(theta2 + theta3)))
#   print("x: "+str(x))
#   print("y: "+str(y))
#   print("------------------------------------")

#   # Convert the angles to degrees and return them
#   return math.degrees(theta1), (180-math.degrees(theta2 + theta3))

def anglecalc(x,y):
    theta2 = np.arccos(((x**2 + y**2 - l1**2 -l2**2)/(2*l1*l2)))

    if x == 0:
        theta1 = np.pi/2 - np.arctan((l2*np.sin(theta2))/(l1 + l2 * np.cos(theta2)))
    else:
        theta1 = np.arctan(y/x) - np.arctan((l2*np.sin(theta2))/(l1 + l2 * np.cos(theta2)))

    print("angle1: "+str(theta1*180/np.pi))
    print("angle2: "+str(theta2*180/np.pi))
    print("x: "+str(x))
    print("y: "+str(y))
    print("------------------------------------")
    return theta1*180/np.pi, theta2*180/np.pi

    
# Declare an named instance of class pass GPIO pins numbers
motor1 = RpiMotorLib.A4988Nema(direction1, step1, GPIO_pins, "A4988")
motor2 = RpiMotorLib.A4988Nema(direction2, step2, GPIO_pins, "A4988")
arm1 = Arm(0,motor1)
arm2 = Arm(0,motor2)


def move_both_arm(angle1, angle2):
  arm1_t = Thread(target=arm1.to_angle, args=(angle1,))
  arm2_t = Thread(target=arm2.to_angle, args=(angle2,))
  arm1_t.start()
  arm2_t.start()
  arm1_t.join()
  arm2_t.join()


def pen_touch(touch):
  if(touch==True):
    lift.ChangeDutyCycle(5) 
    time.sleep(0.1)
  else:
    lift.ChangeDutyCycle(6.5) 
    time.sleep(0.1)


def to_point(x,y):
    angle1, angle2 = anglecalc(x, y)
    move_both_arm(angle1,angle2)

def draw_array(arr, arm1, arm2):
  # for i in range(len(arr)):
  #   if i % 10 ==0 :
  #     print("Cali")
  #     time.sleep(5)
  #     arm1.angle=0
  #     arm2.angle=0
  #     print("Cali-done, cont...")

  for i in range(len(arr)):

    # if(i%5==0):
    #   to_point(8,8)
    #   time.sleep(0.5)
    #   pen_touch(True)
    #   pen_touch(False)
    #   pen_touch(False)

    print("drawing point #:"+str(i))
    to_point(arr[i][0],arr[i][1])

    time.sleep(0.5)

    pen_touch(True)
    pen_touch(False)
    pen_touch(False)



#  r->radius, x y center, n number of points
def make_circle(r, x, y, n):
    angles = np.linspace(0, 2*np.pi, n)
    out = [[x + r*np.cos(theta), y + r*np.sin(theta)] for theta in angles]
    return out

def make_rectangle(w, h, x, y, n):
    xs = np.linspace(x, x + w, n)
    ys = np.linspace(y, y + h, n)
    bottom = [[xi, y] for xi in xs]
    top = [[xi, y + h] for xi in xs]
    left = [[x, yi] for yi in ys]
    right = [[x + w, yi] for yi in ys]
    left.reverse()
    return bottom + right[1:-1] + top[::-1] + left[1:-1]


def factor_arrary(array,x_f, y_f):
  array[:,0]  =array[:,0] * x_f
  array[:,1]  =array[:,1] * y_f
  return array


#LINE DEMO

line_array = []
# for i in range(20):
#   line_array.append([10+offset,5+i*(5/10)])

line_array = [[10,5],[10,10],[7,7],[10,5],[10,10],[7,7],[10,5],[10,10],[7,7],[10,5],[10,10],[7,7],[10,5],[10,10],[7,7],[10,5],[10,10],[7,7],[10,5],[10,10],[7,7]]

# draw_array(line_array,arm1,arm2)

#CIRCLE DEMO

circle  = np.array(make_circle(4,8+offset,5+offset,30))

circle = factor_arrary(circle,0.85,1)
# draw_array(circle,arm1, arm2)


# RECTANGLE DEMO
rect = draw_array(make_rectangle(6,6,offset+2,offset+2,10), arm1, arm2)

draw_array(rect,arm1,arm2)

# IMG
sample_arr=[]
## DOWNSAMPLE
for i in range(np.array(coordinates).shape[0]):
  if i % 30 ==0:
    offsetted  = [coordinates[i][0]+offset,coordinates[i][1]+offset]
    sample_arr.append(offsetted)

print(sample_arr)
sample_arr= np.array(sample_arr)

factor_arrary(sample_arr,0.85,1)
# draw_array(sample_arr, arm1, arm2)



####ENDING#####
to_point(12,0)
pen_touch(False)

lift.stop()
GPIO.cleanup()