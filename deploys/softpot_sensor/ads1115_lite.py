#!/usr/bin/env python3
"""
ads1115_lite.py

Minimal, dependency-free ADS1115 driver for Linux / Raspberry Pi.
Talks to the chip directly over /dev/i2c-1 via ioctl — no adafruit-blinka,
no smbus/smbus2, nothing outside the Python standard library. Useful when
the board has no internet access to pip/apt install anything.

Requires:
    - I2C enabled (sudo raspi-config -> Interface Options -> I2C)
    - Your user able to access /dev/i2c-1 (usually the 'i2c' group,
      already true if i2cdetect -y 1 worked for you)

Usage:
    from ads1115_lite import ADS1115

    with ADS1115(address=0x48) as ads:
        voltage = ads.read_voltage(channel=0, gain=1)
"""
import os
import fcntl
import struct
import time

I2C_SLAVE = 0x0703  # Linux ioctl request code to set the target I2C slave address

_CONFIG_REG = 0x01
_CONVERSION_REG = 0x00

# Same gain convention as adafruit-circuitpython-ads1x15: gain -> full-scale range (V)
_GAIN_TO_FS = {
    2 / 3: 6.144,
    1: 4.096,
    2: 2.048,
    4: 1.024,
    8: 0.512,
    16: 0.256,
}
_GAIN_TO_BITS = {
    2 / 3: 0x0000,
    1: 0x0200,
    2: 0x0400,
    4: 0x0600,
    8: 0x0800,
    16: 0x0A00,
}

_OS_START = 0x8000
_MUX_SINGLE_ENDED_BASE = 0x4000  # AIN0 single-ended = 0b100 << 12
_MODE_SINGLE_SHOT = 0x0100
_DR_128SPS = 0x0080              # 128 samples/sec — plenty for a slider
_COMP_QUE_DISABLE = 0x0003       # disable comparator (required, ALERT/RDY unused)


class ADS1115:
    def __init__(self, address=0x48, bus=1):
        self.address = address
        self._fd = os.open(f'/dev/i2c-{bus}', os.O_RDWR)
        fcntl.ioctl(self._fd, I2C_SLAVE, self.address)

    def close(self):
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def read_voltage(self, channel=0, gain=1):
        """Read a single-ended voltage (V) from AIN0-AIN3."""
        if channel not in (0, 1, 2, 3):
            raise ValueError('channel must be 0, 1, 2, or 3')
        if gain not in _GAIN_TO_FS:
            raise ValueError(f'gain must be one of {list(_GAIN_TO_FS)}')

        mux = _MUX_SINGLE_ENDED_BASE + (channel << 12)
        config = (
            _OS_START
            | mux
            | _GAIN_TO_BITS[gain]
            | _MODE_SINGLE_SHOT
            | _DR_128SPS
            | _COMP_QUE_DISABLE
        )

        # Write config register (register pointer + 2 data bytes, MSB first)
        os.write(self._fd, bytes([_CONFIG_REG, (config >> 8) & 0xFF, config & 0xFF]))

        # 128 SPS -> ~7.8ms/sample; wait a little longer to be safe
        time.sleep(0.009)

        # Point at the conversion register, then read the 2-byte result
        os.write(self._fd, bytes([_CONVERSION_REG]))
        raw_bytes = os.read(self._fd, 2)
        raw = struct.unpack('>h', raw_bytes)[0]  # signed, MSB first

        fs = _GAIN_TO_FS[gain]
        return raw * fs / 32768.0
