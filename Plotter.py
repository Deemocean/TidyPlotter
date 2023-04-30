import RPi.GPIO as GPIO
import time

A1= 4
B1= 17
C1= 27
D1= 22

A2= 5
B2= 6
C2= 13
D2= 19

T=0.005

h = GPIO.HIGH
l = GPIO.LOW


GPIO.setmode(GPIO.BCM)
GPIO.setup(A1, GPIO.OUT)
GPIO.setup(B1, GPIO.OUT)
GPIO.setup(C1, GPIO.OUT) 
GPIO.setup(D1, GPIO.OUT)

GPIO.setup(A2, GPIO.OUT)
GPIO.setup(B2, GPIO.OUT) 
GPIO.setup(C2, GPIO.OUT)
GPIO.setup(D2, GPIO.OUT)

def setMotor(motor, w1,w2,w3,w4):
	if motor == 0:
		GPIO.output(A1,w1)
		GPIO.output(B1,w2)
		GPIO.output(C1,w3)
		GPIO.output(D1,w4)
	elif motor==1:
		GPIO.output(A2,w1)
		GPIO.output(B2,w2)
		GPIO.output(C2,w3)
		GPIO.output(D2,w4)
		
#11.25 degree
def forward(motor):
	setMotor(motor,h,l,l,h)
	time.sleep(T)
	setMotor(motor,h,h,l,l)
	time.sleep(T)
	setMotor(motor,l,h,h,l)
	time.sleep(T)
	setMotor(motor,l,l,h,h)
	time.sleep(T)

def reverse(motor):
	setMotor(motor,h,l,l,h)
	time.sleep(T)
	setMotor(motor,l,l,h,h)
	time.sleep(T)
	setMotor(motor,l,h,h,l)
	time.sleep(T)
	setMotor(motor,h,h,l,l)
	time.sleep(T)
	

def test():
	for i in range(256):
		forward(0)
		forward(1)
	for i in range(256):
		reverse(0)
		reverse(1)
	
	
	

test()

    

	

	
		
