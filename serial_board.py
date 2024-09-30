import serial, time


def is_board_ready():
    for p, rate in [('COM3', 9600), ('COM4', 115200)]:
        try:
            ser = serial.Serial(p, rate, timeout=1)
            ser.close()
        except (OSError, serial.SerialException):
            return False
    return True


class SerialBoard:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def make_command(self, ctrl_word: int, cmd_word: int, data: int = 0x0f):
        result = bytearray([0x53, 0x59, ctrl_word, cmd_word, 0x00, 0x01])
        result.append(data)
        checksum = sum(result[:7]) & 0xFF
        result.append(checksum)
        result.extend([0x54, 0x43])
        return result

    def read_data(self, data_len=1):
        start_time = time.time()
        while time.time() - start_time < 1:
            data = self.ser.read(2)
            if data == b'\x53\x59':
                remaining_data = self.ser.read(7 + data_len)
                if (remaining_data[:2] == b'\x01\x01') or (remaining_data[:2] == b'\x81\x0b'):
                    continue
                full_data = data + remaining_data
                #print(full_data)
                if (len(full_data) == 9 + data_len) and (full_data[-2:] == b'\x54\x43'):
                    return full_data[6:6+data_len]
        return None

    def close(self):
        self.ser.close()

class SleepBoard(SerialBoard):
    def __init__(self, port='COM4', baudrate=115200):
        super().__init__(port, baudrate)
        ## Not use
        #self.move_auto_off_cmd = self.make_command(0x80, 0x00, 0x00)
        #self.move_auto_on_cmd = self.make_command(0x80, 0x00, 0x01)
        #self.heart_auto_off_cmd = self.make_command(0x85, 0x00, 0x00)
        #self.heart_auto_on_cmd = self.make_command(0x85, 0x00, 0x01)
        #self.breath_auto_off_cmd = self.make_command(0x81, 0x00, 0x00)
        #self.breath_auto_on_cmd = self.make_command(0x81, 0x00, 0x01)
        self.commands = {
        "move_state" : self.make_command(0x80, 0x82),
        "move_value" : self.make_command(0x80, 0x83),
        "heart_rate" : self.make_command(0x85, 0x82),
        "breath_rate" : self.make_command(0x81, 0x82),
        "sleep_state" : self.make_command(0x84, 0x82)
        }

    # #Not use
    #def set_auto_monitoring(self, feature, state):
    #    cmd = getattr(self, f'{feature}_auto_{"on" if state else "off"}_cmd')
    #    self.ser.write(cmd)

    def get_value(self, data_name):
        self.ser.write(self.commands[data_name])
        value = self.read_data()
        if value is not None:
            return int(value.hex(), 16)
        return None

class ControlBoard(SerialBoard):
    def __init__(self, port='COM3', baudrate=9600):
        super().__init__(port, baudrate)
        self.commands = {
        "sound_value" : self.make_command(0x20, 0x82),
        "temp_value" : self.make_command(0x20, 0x83)        
        }

    def get_value(self, data_name):
        self.ser.write(self.commands[data_name])
        value = self.read_data(data_len=2)
        if value is not None:
            return int(value.hex(), 16)
        return None
