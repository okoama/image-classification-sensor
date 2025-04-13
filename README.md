# image-classification-sensor
Real-time image classification and proximity alert using TensorFlow Lite and Raspberry Pi
# Real-Time Image Classification and Proximity Alert System

This project combines **computer vision** and **ultrasonic sensing** to create a real-time object detection system with **audible alerts** using a **Raspberry Pi**, a **USB camera**, and **TensorFlow Lite**.

---

## 🔧 Features

- 📸 Real-time image classification using TensorFlow Lite
- 🗣️ Audible readout of detected object names using `espeak`
- 📏 Ultrasonic sensor-based distance detection
- 🚨 Voice warning when an object is too close

---

## 🧰 Requirements

- Raspberry Pi (with GPIO)
- USB Camera
- HC-SR04 Ultrasonic Sensor
- Python 3
- TensorFlow Lite runtime
- `espeak` text-to-speech engine

---

## 📦 Installation

```bash
sudo apt update
sudo apt install python3-pip espeak
pip3 install opencv-python numpy
