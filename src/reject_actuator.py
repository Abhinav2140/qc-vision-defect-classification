"""
reject_actuator.py — Interface between the inspection software and the
physical mechanism that removes a defective part from the line.

Ships with a SimulatedActuator (safe default, logs actions only) plus two
common real-world backends stubbed out for you to wire up:

  - ModbusActuator: for PLCs (Siemens S7, Allen-Bradley, etc.) that expose
    a Modbus TCP coil for the reject solenoid/pusher/air-blast. Most
    automotive and electronics lines already run Modbus or Ethernet/IP.
  - GPIOActuator: for a Raspberry Pi / Jetson driving a relay directly
    (common in smaller textile or pharma packaging retrofits without a PLC).

IMPORTANT — safety: the vision system should request a reject; a
PLC-side safety interlock (e.g. confirming the part is in the reject
window, e-stop status, line speed) should have final authority to fire
the actuator. Never wire a camera's inference result directly to power
the reject mechanism without a PLC/safety-relay in between.
"""

import time
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reject_actuator")


class RejectActuator(ABC):
    @abstractmethod
    def trigger_reject(self, part_id: str, defect_type: str, severity: float) -> bool:
        """Fire the reject mechanism. Returns True if the command was
        acknowledged by the hardware/PLC."""
        ...


class SimulatedActuator(RejectActuator):
    """Default backend — logs the action instead of firing real hardware.
    Use this in dev/staging, and swap to ModbusActuator/GPIOActuator only
    once the line's controls engineer has validated wiring and timing."""

    def __init__(self):
        self.reject_log = []

    def trigger_reject(self, part_id: str, defect_type: str, severity: float) -> bool:
        event = {"part_id": part_id, "defect_type": defect_type,
                 "severity": severity, "ts": time.time()}
        self.reject_log.append(event)
        logger.info(f"[SIMULATED REJECT] part={part_id} defect={defect_type} "
                    f"severity={severity:.2f}")
        return True


class ModbusActuator(RejectActuator):
    """Writes a coil on a PLC over Modbus TCP to fire a pusher/air-blast/
    diverter. Requires `pip install pymodbus`.

    Typical wiring: the vision PC computes the decision, then pulses a
    coil that the PLC's ladder logic reads; the PLC handles the actual
    timing against encoder position so the reject fires when the
    defective part physically reaches the reject station (accounting
    for conveyor transport delay from camera to reject point)."""

    def __init__(self, host: str, port: int = 502, coil_address: int = 0,
                 transport_delay_s: float = 0.0):
        self.host, self.port = host, port
        self.coil_address = coil_address
        self.transport_delay_s = transport_delay_s
        self._client = None  # lazily connect

    def _connect(self):
        from pymodbus.client import ModbusTcpClient
        self._client = ModbusTcpClient(self.host, port=self.port)
        self._client.connect()

    def trigger_reject(self, part_id: str, defect_type: str, severity: float) -> bool:
        if self._client is None:
            self._connect()
        if self.transport_delay_s > 0:
            # Schedule rather than block in production — shown inline here
            # for clarity. Use a timer thread or the PLC's own encoder-based
            # delay logic instead of a blocking sleep on the vision PC.
            time.sleep(self.transport_delay_s)
        result = self._client.write_coil(self.coil_address, True)
        logger.info(f"[MODBUS REJECT] part={part_id} defect={defect_type} "
                    f"coil={self.coil_address} ok={not result.isError()}")
        return not result.isError()


class GPIOActuator(RejectActuator):
    """Drives a relay pin directly from an edge device (Raspberry Pi /
    Jetson). Requires `pip install RPi.GPIO` (or Jetson.GPIO on Jetson)."""

    def __init__(self, pin: int, pulse_ms: int = 150):
        self.pin = pin
        self.pulse_ms = pulse_ms
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def trigger_reject(self, part_id: str, defect_type: str, severity: float) -> bool:
        self.GPIO.output(self.pin, self.GPIO.HIGH)
        time.sleep(self.pulse_ms / 1000)
        self.GPIO.output(self.pin, self.GPIO.LOW)
        logger.info(f"[GPIO REJECT] part={part_id} defect={defect_type} pin={self.pin}")
        return True
