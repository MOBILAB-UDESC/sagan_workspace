import smbus   # not smbus2, just smbus
import time
import statistics

bus = smbus.SMBus(1)

# ── addresses from your code ──────────────────────────────
RIGHT_PICO = 0x16
LEFT_PICO  = 0x15

# ── commands from your code ───────────────────────────────
CMD_SET_VELOCITY = 0xA1    # master -> slave  (ControlData)
CMD_GET_SENSORS  = 0xB1    # slave  -> master (SensorData)

# ── struct sizes from your code ───────────────────────────
# ControlData: cmd(1) + front_velocity(2) + rear_velocity(2) = 5 bytes
# SensorData:  front_vel(2) + front_cur(2) + rear_vel(2) + rear_cur(2) = 8 bytes

DATA_SCALE_FACTOR = 100.0
NUM_SAMPLES       = 500

# ─────────────────────────────────────────────────────────
def pack_control_data(front_vel: float, rear_vel: float) -> list:
    """Pack ControlData struct into bytes"""
    front_raw = int(front_vel * DATA_SCALE_FACTOR)
    rear_raw  = int(rear_vel  * DATA_SCALE_FACTOR)
    return [
        CMD_SET_VELOCITY,
        (front_raw >> 8) & 0xFF, front_raw & 0xFF,   # int16 big-endian
        (rear_raw  >> 8) & 0xFF, rear_raw  & 0xFF
    ]

def unpack_sensor_data(data: list) -> dict:
    """Unpack SensorData struct from bytes"""
    def to_int16(hi, lo):
        val = (hi << 8) | lo
        return val - 65536 if val > 32767 else val

    return {
        "front_velocity": to_int16(data[0], data[1]) / DATA_SCALE_FACTOR,
        "front_current":  to_int16(data[2], data[3]) / 1000.0,
        "rear_velocity":  to_int16(data[4], data[5]) / DATA_SCALE_FACTOR,
        "rear_current":   to_int16(data[6], data[7]) / 1000.0,
    }

# ─────────────────────────────────────────────────────────
def test_write_time(address, n=NUM_SAMPLES):
    """Test CMD_SET_VELOCITY write time"""
    times = []
    payload = pack_control_data(10.0, 10.0)

    for _ in range(n):
        start = time.perf_counter()
        bus.write_i2c_block_data(address, CMD_SET_VELOCITY, payload[1:])
        end = time.perf_counter()
        times.append((end - start) * 1000)

    print(f"\n── WRITE test (0xA1) → addr 0x{address:02X} ──────────────")
    print(f"  Samples : {n}")
    print(f"  Average : {statistics.mean(times):.3f} ms")
    print(f"  Min     : {min(times):.3f} ms")
    print(f"  Max     : {max(times):.3f} ms")
    print(f"  Std dev : {statistics.stdev(times):.3f} ms")
    print(f"  Max freq: {1000/statistics.mean(times):.1f} Hz")

def test_read_time(address, n=NUM_SAMPLES):
    """Test CMD_GET_SENSORS read time"""
    times = []

    for _ in range(n):
        start = time.perf_counter()
        data  = bus.read_i2c_block_data(address, CMD_GET_SENSORS, 8)
        end   = time.perf_counter()
        times.append((end - start) * 1000)

    print(f"\n── READ test (0xB1) → addr 0x{address:02X} ───────────────")
    print(f"  Samples : {n}")
    print(f"  Average : {statistics.mean(times):.3f} ms")
    print(f"  Min     : {min(times):.3f} ms")
    print(f"  Max     : {max(times):.3f} ms")
    print(f"  Std dev : {statistics.stdev(times):.3f} ms")
    print(f"  Max freq: {1000/statistics.mean(times):.1f} Hz")

def test_full_cycle(address, n=NUM_SAMPLES):
    """Test full write + read cycle time (what your ROS2 node actually does)"""
    times = []
    payload = pack_control_data(10.0, 10.0)

    for _ in range(n):
        start = time.perf_counter()

        # write velocities
        bus.write_i2c_block_data(address, CMD_SET_VELOCITY, payload[1:])
        # read sensor data back
        data = bus.read_i2c_block_data(address, CMD_GET_SENSORS, 8)

        end = time.perf_counter()
        times.append((end - start) * 1000)

    sensors = unpack_sensor_data(data)

    print(f"\n── FULL CYCLE test → addr 0x{address:02X} ────────────────")
    print(f"  Samples : {n}")
    print(f"  Average : {statistics.mean(times):.3f} ms")
    print(f"  Min     : {min(times):.3f} ms")
    print(f"  Max     : {max(times):.3f} ms")
    print(f"  Std dev : {statistics.stdev(times):.3f} ms")
    print(f"  Max freq: {1000/statistics.mean(times):.1f} Hz")
    print(f"  Last sensor reading: {sensors}")

def test_both_picos(n=NUM_SAMPLES):
    """Test reading from both Picos sequentially (as ros2_control does)"""
    times = []
    payload = pack_control_data(10.0, 10.0)

    for _ in range(n):
        start = time.perf_counter()

        # left pico
        bus.write_i2c_block_data(LEFT_PICO,  CMD_SET_VELOCITY, payload[1:])
        bus.read_i2c_block_data (LEFT_PICO,  CMD_GET_SENSORS,  8)

        # right pico
        bus.write_i2c_block_data(RIGHT_PICO, CMD_SET_VELOCITY, payload[1:])
        bus.read_i2c_block_data (RIGHT_PICO, CMD_GET_SENSORS,  8)

        end = time.perf_counter()
        times.append((end - start) * 1000)

    print(f"\n── BOTH PICOS cycle ─────────────────────────────────────")
    print(f"  Samples : {n}")
    print(f"  Average : {statistics.mean(times):.3f} ms")
    print(f"  Min     : {min(times):.3f} ms")
    print(f"  Max     : {max(times):.3f} ms")
    print(f"  Std dev : {statistics.stdev(times):.3f} ms")
    print(f"  Max freq: {1000/statistics.mean(times):.1f} Hz")
    print(f"  → This is your ros2_control real max rate")

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("I2C timing test for Sagan Pico slaves")
    print(f"Testing address 0x{RIGHT_PICO:02X}...")

    test_write_time(RIGHT_PICO)
    test_read_time(RIGHT_PICO)
    test_full_cycle(RIGHT_PICO)
    test_both_picos()

    bus.close()