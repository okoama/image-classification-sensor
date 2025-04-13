import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import RPi.GPIO as GPIO


warning = "Too Close!"


# Intialize sensor values
GPIO.setmode(GPIO.BCM)
TRIG = 23
ECHO = 24
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)


# Load labels and save to list
open_labels = open("/home/okoliama/Downloads/labels.txt", "r")
read_labels = open_labels.read()
labels = read_labels.split("\n")


# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="/home/okoliama/Downloads/saved_model.tflite")
interpreter.allocate_tensors()


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
________________


# Open a video capture object with the index corresponding to the USB camera.
cap = cv2.VideoCapture(0)


# Set the camera resolution (adjust as needed).
cap.set(3, 640)
cap.set(4, 480)


while True:
    # Capture a frame from the USB camera.
    ret, frame = cap.read()


    # Resize the frame to match the input shape expected by the model.
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))


    # Preprocess the input image (normalize and reshape).
    input_data = resized_frame.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)


    # Set the input tensor for the model.
    interpreter.set_tensor(input_details[0]['index'], input_data)


    # Run inference.
    interpreter.invoke()


    # Get the output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])


    # Post-process the output (get the predicted class and confidence).
    predicted_class = np.argmax(output_data)
    class_name = labels[predicted_class]
    confidence = output_data[0, predicted_class]
________________


    # Display the classification result on the frame.
    cv2.putText(frame, f"Class: {class_name}, Confidence: {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Read out class prediction
    os.system('espeak "'+class_name+'"')


    # Display the frame.
    cv2.imshow('Real-time Image Classification', frame)
    
    # Sensor
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)


    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start


    distance = pulse_duration * 17150


    distance = round(distance, 2)
    
    if distance <= 10:
        os.system('espeak "'+warning+'"')


    # Break the loop if 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture object and close the window.
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()


Inferences
  # Run inference.
    interpreter.invoke()


    # Get the output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])


    # Post-process the output (get the predicted class and confidence).
    predicted_class = np.argmax(output_data)
    class_name = labels[predicted_class]
    confidence = output_data[0, predicted_class]




Sensor
   GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)


    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start


    distance = pulse_duration * 17150


    distance = round(distance, 2)
    
    if distance <= 10:
        os.system('espeak "'+warning+'"')


Tts
  
    # Read out class prediction
    os.system('espeak "'+class_name+'"')


Libraries
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import RPi.GPIO as GPIO