from ads1115_lite import ADS1115

class SoftPot:
    
    def __init__(self):
        self.ads = ADS1115(address=0x48)
        self.history = []

    #returns voltage
    def read_voltage(self):
        return self.ads.read_voltage(channel=0)

    #returns position
    def read_percent(self):
        IDLE_VOLTAGE = 0.9 #voltage when idle
        TOUCH_THRESHOLD = 0.03  # 30 mV
        MIN_V = 0.01 # minimum voltage
        MAX_V = 3.3 # maximum voltage
        NO_TOUCH = 0.10  # adjust after testing
        
        value = self.ads.read_voltage(channel=0)

        # moving average filter
        self.history.append(value)
        if len(self.history) > 20:
            self.history.pop(0)
    
        value = sum(self.history) / len(self.history)

        if value < NO_TOUCH:
            return None
        # Ignore readings near the idle voltage
        if abs(value - IDLE_VOLTAGE) < TOUCH_THRESHOLD:
            return None
    
        p = (value - MIN_V) / (MAX_V - MIN_V)
        return max(0.0, min(1.0, p))