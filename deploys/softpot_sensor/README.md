# softpot_sensor

Softpot Class for a 500mm SoftPot linear position sensor,
read through an ADS1115 I2C ADC on a Raspberry Pi.

## 1. Wiring

| SoftPot | -> | Target                |
|---------|----|-----------------------|
| End A   | -> | Pi 3.3V (pin 17)      |
| End B   | -> | Pi GND (pin 30)       |
| Wiper   | -> | ADS1115 A0            |

| ADS1115 | -> | Target                      |
|---------|----|-----------------------------|
| VDD     | -> | Pi 3.3V (pin 1)             |
| GND     | -> | Pi GND (pin 9)              |
| SCL     | -> | Pi GPIO3 (pin 5)            |
| SDA     | -> | Pi GPIO2 / (pin 3)          |
| ADDR    | -> | GND (address 0x48) (pin 39) |

## 2. One-time Pi Setup

```bash
sudo raspi-config          # Interface Options -> I2C -> Enable, then reboot
# verify the ADS1115 shows up at 0x48
i2cdetect -y 1
```

## 3. Import Softpot Class & ADS

```bash
from ads1115_lite import ADS1115
from softpot import SoftPot
```

Calibrate:
Press each physical end of the strip, note the two steady voltages, and
use them as `MIN_V` / `MAX_V` below.

## 4. Functions

```bash
def __init__(self):
#initializes ADS1115
def read_voltage(self):
#returns voltage (0.1 -> 3.3)
def read_percent(self):
#returns position (0.0 -> 1.0)
```

## 5. Run

```bash
python3 _______.py
```