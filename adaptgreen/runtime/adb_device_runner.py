"""
AdaptGreen ADB Device Runner & Real Telemetry Bridge
Connects to a physical Android device via ADB to extract real hardware metrics:
- Real battery current and voltage (instantaneous power mW and energy mJ)
- Real SoC die / junction temperature from Linux thermal_zone sysfs nodes
- Real CPU core frequencies to observe dynamic DVFS thermal throttling
- Real available RAM and Wi-Fi RSSI signal strength
"""

import re
import shutil
import logging
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..cme.simulator import DeviceState

logger = logging.getLogger("AdaptGreen.ADBBridge")


@dataclass
class AndroidHardwareTelemetry:
    """Snapshot of raw physical hardware metrics from an Android device."""
    connected: bool
    device_model: str
    soc_hardware: str
    battery_level_pct: float
    battery_voltage_mv: float
    battery_current_ma: float
    power_mw: float
    soc_temp_c: float
    available_ram_mb: float
    wifi_rssi_dbm: float
    cpu_frequencies_mhz: List[float]
    is_charging: bool
    status_note: str


class ADBDeviceRunner:
    """
    Bridge communicating with a connected Android smartphone via ADB.
    """

    def __init__(self, device_serial: Optional[str] = None):
        self.device_serial = device_serial
        self.adb_bin = shutil.which("adb")
        self.device_info: Dict[str, str] = {}
        if self.is_adb_available():
            self._init_device()

    def is_adb_available(self) -> bool:
        """Checks if ADB executable exists in PATH."""
        return self.adb_bin is not None

    @classmethod
    def detect_devices(cls) -> List[str]:
        """Returns list of connected Android device serials."""
        adb_bin = shutil.which("adb")
        if not adb_bin:
            return []
        try:
            res = subprocess.run([adb_bin, "devices"], capture_output=True, text=True, timeout=4)
            lines = res.stdout.strip().splitlines()
            devices = []
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
            return devices
        except Exception as e:
            logger.warning("Error detecting ADB devices: %s", e)
            return []

    def _run_shell(self, cmd: str, timeout: int = 4) -> str:
        """Executes a command inside the Android shell via ADB."""
        if not self.adb_bin:
            return ""
        args = [self.adb_bin]
        if self.device_serial:
            args.extend(["-s", self.device_serial])
        args.extend(["shell", cmd])
        try:
            res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            return res.stdout.strip()
        except Exception as e:
            logger.debug("ADB command '%s' failed: %s", cmd, e)
            return ""

    def _init_device(self) -> None:
        """Retrieves static device hardware properties."""
        model = self._run_shell("getprop ro.product.model") or "Unknown Android Device"
        brand = self._run_shell("getprop ro.product.brand") or ""
        soc = self._run_shell("getprop ro.hardware") or self._run_shell("getprop ro.board.platform") or "ARM SoC"
        android_ver = self._run_shell("getprop ro.build.version.release") or "Unknown"
        self.device_info = {
            "model": f"{brand.capitalize()} {model}".strip(),
            "soc": soc,
            "android_version": android_ver
        }

    def get_real_telemetry(self) -> AndroidHardwareTelemetry:
        """
        Polls physical Android hardware metrics directly from kernel sysfs.
        """
        if not self.is_adb_available():
            return AndroidHardwareTelemetry(
                connected=False,
                device_model="None (ADB not installed)",
                soc_hardware="N/A",
                battery_level_pct=100.0,
                battery_voltage_mv=4000.0,
                battery_current_ma=0.0,
                power_mw=0.0,
                soc_temp_c=35.0,
                available_ram_mb=4000.0,
                wifi_rssi_dbm=-50.0,
                cpu_frequencies_mhz=[],
                is_charging=False,
                status_note="ADB is not installed or not in PATH."
            )

        devices = self.detect_devices()
        if not devices:
            return AndroidHardwareTelemetry(
                connected=False,
                device_model="None (No device detected)",
                soc_hardware="N/A",
                battery_level_pct=100.0,
                battery_voltage_mv=4000.0,
                battery_current_ma=0.0,
                power_mw=0.0,
                soc_temp_c=35.0,
                available_ram_mb=4000.0,
                wifi_rssi_dbm=-50.0,
                cpu_frequencies_mhz=[],
                is_charging=False,
                status_note="No Android device connected over USB. Enable USB Debugging in Developer Options."
            )

        # 1. Real Battery Power & Current
        # Try /sys/class/power_supply/battery nodes
        curr_raw = self._run_shell("cat /sys/class/power_supply/battery/current_now 2>/dev/null")
        volt_raw = self._run_shell("cat /sys/class/power_supply/battery/voltage_now 2>/dev/null")
        batt_dump = self._run_shell("dumpsys battery 2>/dev/null")

        battery_level = 100.0
        is_charging = False
        if batt_dump:
            lvl_match = re.search(r"level:\s*(\d+)", batt_dump)
            if lvl_match:
                battery_level = float(lvl_match.group(1))
            stat_match = re.search(r"status:\s*(\d+)", batt_dump)
            if stat_match and stat_match.group(1) == "2":
                is_charging = True

        # Parse current (often in microamperes)
        current_ma = 0.0
        if curr_raw and curr_raw.lstrip("-").isdigit():
            c_val = float(curr_raw)
            current_ma = abs(c_val) / 1000.0 if abs(c_val) > 10000 else abs(c_val)

        # Parse voltage (often in microvolts)
        voltage_mv = 4000.0
        if volt_raw and volt_raw.isdigit():
            v_val = float(volt_raw)
            voltage_mv = v_val / 1000.0 if v_val > 10000 else v_val

        # Instantaneous power in milliwatts: P = (V_mV * I_mA) / 1000.0
        power_mw = (voltage_mv * current_ma) / 1000.0 if current_ma > 0 else 1200.0

        # 2. Real SoC Temperature
        # Look across /sys/class/thermal/thermal_zone*
        temp_c = 38.0
        tz_output = self._run_shell(
            "for tz in /sys/class/thermal/thermal_zone*; do "
            "  echo \"$(cat $tz/type 2>/dev/null):$(cat $tz/temp 2>/dev/null)\"; "
            "done"
        )
        if tz_output:
            temps = []
            for line in tz_output.splitlines():
                if ":" in line:
                    t_type, t_val = line.split(":", 1)
                    if t_val.isdigit():
                        v = float(t_val)
                        v_c = v / 1000.0 if v > 1000 else v
                        # prioritize CPU/SoC thermal zones
                        if any(k in t_type.lower() for k in ["cpu", "soc", "tsens", "cluster"]):
                            temps.append(v_c)
            if temps:
                temp_c = max(temps)

        # 3. Real Available RAM
        avail_ram_mb = 3500.0
        meminfo = self._run_shell("cat /proc/meminfo 2>/dev/null")
        if meminfo:
            mem_match = re.search(r"MemAvailable:\s*(\d+)\s*kB", meminfo)
            if mem_match:
                avail_ram_mb = float(mem_match.group(1)) / 1024.0

        # 4. Real Wi-Fi RSSI
        wifi_rssi = -55.0
        wifi_dump = self._run_shell("dumpsys wifi 2>/dev/null | grep -i 'rssi'")
        if wifi_dump:
            rssi_match = re.search(r"rssi\s*=?\s*(-?\d+)", wifi_dump, re.IGNORECASE)
            if rssi_match:
                wifi_rssi = float(rssi_match.group(1))

        # 5. Real CPU Core Frequencies (DVFS state)
        freqs_mhz = []
        freq_out = self._run_shell(
            "for cpu in /sys/devices/system/cpu/cpu[0-9]*; do "
            "  cat $cpu/cpufreq/scaling_cur_freq 2>/dev/null; "
            "done"
        )
        if freq_out:
            for f in freq_out.splitlines():
                if f.strip().isdigit():
                    freqs_mhz.append(round(float(f) / 1000.0, 1))

        return AndroidHardwareTelemetry(
            connected=True,
            device_model=self.device_info.get("model", "Android Device"),
            soc_hardware=self.device_info.get("soc", "ARM SoC"),
            battery_level_pct=battery_level,
            battery_voltage_mv=voltage_mv,
            battery_current_ma=current_ma,
            power_mw=power_mw,
            soc_temp_c=temp_c,
            available_ram_mb=avail_ram_mb,
            wifi_rssi_dbm=wifi_rssi,
            cpu_frequencies_mhz=freqs_mhz,
            is_charging=is_charging,
            status_note=f"Live telemetry from {self.device_info.get('model', 'Android Device')} (Android {self.device_info.get('android_version', '?')})"
        )

    def to_device_state(self, telemetry: Optional[AndroidHardwareTelemetry] = None) -> DeviceState:
        """
        Converts real physical Android hardware readings into an AdaptGreen DeviceState.
        """
        telem = telemetry or self.get_real_telemetry()
        return DeviceState(
            battery=telem.battery_level_pct,
            temperature=telem.soc_temp_c,
            rssi=telem.wifi_rssi_dbm,
            user_activity=1,  # Foreground by default when interacting
            task_type=1,       # Recommendation task
            available_ram=telem.available_ram_mb
        )
