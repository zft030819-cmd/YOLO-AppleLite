"""Read-only Linux CPU/process sampling for identifying periodic interference."""
import json
import os
from pathlib import Path
import platform
import threading
import time


class Monitor:
    def __init__(self, path, start, context, interval=1):
        self.path, self.start, self.context, self.interval = path, start, context, interval
        self.stop = threading.Event()
        self.thread = None

    def begin(self):
        if platform.system() == 'Linux':
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)

    def run(self):
        previous, old_total, old_idle = {}, None, None
        with self.path.open('w', encoding='utf-8') as output:
            while not self.stop.is_set():
                try:
                    ticks = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
                    total, idle = sum(ticks), ticks[3] + ticks[4]
                    delta = total - old_total if old_total is not None else 0
                    processes, current = [], {}
                    for directory in Path('/proc').iterdir():
                        if not directory.name.isdecimal():
                            continue
                        try:
                            raw = (directory / 'stat').read_text()
                            command = raw[raw.find('(') + 1:raw.rfind(')')]
                            values = raw[raw.rfind(')') + 2:].split()
                            cpu = int(values[11]) + int(values[12])
                            key = (directory.name, values[19])
                            current[key] = cpu
                            percent = 100 * (cpu - previous[key]) * (os.cpu_count() or 1) / delta if delta > 0 and key in previous else 0
                            if percent > .1:
                                processes.append(dict(pid=int(directory.name), command=command, cpu_percent=round(percent, 2)))
                        except (OSError, ValueError, IndexError):
                            continue
                    record = dict(elapsed_s=round(time.perf_counter() - self.start, 3), context=self.context.copy(),
                                  cpu_busy_percent=round(100 * (1 - (idle - old_idle) / delta), 2) if delta > 0 else None,
                                  top_processes=sorted(processes, key=lambda p: -p['cpu_percent'])[:8])
                    for name, path in [('temperature_millicelsius', '/sys/class/thermal/thermal_zone0/temp'),
                                       ('cpu0_frequency_khz', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')]:
                        try:
                            record[name] = int(Path(path).read_text())
                        except (OSError, ValueError):
                            pass
                    output.write(json.dumps(record) + '\n')
                    output.flush()
                    previous, old_total, old_idle = current, total, idle
                except OSError as exc:
                    output.write(json.dumps(dict(error=str(exc))) + '\n')
                    output.flush()
                self.stop.wait(self.interval)
