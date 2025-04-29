# Terktronix-2024B-Oscilloscope-GUI


## Usage

~~~ shell
pip install ./requirements.txt
~~~

~~~ python
python pyqt_gui.py
~~~

# feature
## signal capture
- Generate Simulation signal for  testing
- reading voltage or current signal from Arduino analog port
- reading data from Terktronix 2024B Oscilloscope or Oscilloscope with USB interface
## Nanowire controller
- Nanowire controller communication with AWG
- Linear control feature for nanowires
- PID control feature for nanowires achieve ultra precision < 50nm

# Main 
![](https://lskypro.bin-lian.com/i/2025/03/01/67c297d56a493.jpg)

## Signal capture feature


![](https://lskypro.bin-lian.com/i/2025/03/01/67c298b7bee07.png)

- Real time voltage capture and display
- 4 Channels support could open and close at anytime
- Acuisition time for the refreshing rate
- Well tuned signal saving feature
![](https://lskypro.bin-lian.com/i/2025/03/01/67c299ede4271.jpg)
![](https://lskypro.bin-lian.com/i/2025/03/01/67c29a1854604.png)

# also nanowire control feature

## Nanowire control panel
![](https://lskypro.bin-lian.com/i/2025/03/01/67c29e2ad51fe.png)

## PID controller with target path planning and visualize in real time
![](https://lskypro.bin-lian.com/i/2025/03/01/67c29bf0cd518.png)

### 2025/02/25
- ADD ARDUINO READ FEATURE

![](https://lskypro.bin-lian.com/i/2025/02/25/67bd6268d8b14.png)


with proper Arduino nano or better board and RC low pass circuits, confident measure signal change < 10ms wit smooth signal detection

## 2025/02/26

Solve the Nanowire controller Channel Updating bug -- A recursion problem when I wrote the code.

Solve the Nanowire controller threshold Updating bug

Solve the issue communicating with nanocontroller AWG


Solve bug on linuxmint the return in linux mint is different from mac use '\r' instead of '\n'

## 2025/02/27

adding the nanowire controller PID controller features
- Canvas for PID status monitoring
- real time PID Control

the Minor bug is still exist, but the basic PID control is working now.

## 2025/03/03

adding the voltage speed mapping feature

![](https://lskypro.bin-lian.com/i/2025/03/04/67c61ff52ba73.png)