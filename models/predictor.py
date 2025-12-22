# predictor.py
"""
Execution time predictor using a simple Roofline-like mathematical model.

Model design:
- Compute-bound time:  F / (F_peak * eff_comp)
- Memory-bound time: M / (B_peak * eff_mem)
- Take max(compute_time, memory_time) as base (roofline idea)
- Apply parallelism factor: divide by effective parallelism (P_effective)
- Apply specialization factor: tasks with higher S may run faster on specialized hardware
- Add device latency (startup overhead)
- Multiply by calibration_factor to account for observed bias (learned online in update_model)

Predictor supports:
- predict_execution_time(task, device): returns float seconds (deterministic)
- simulate_actual_time(predicted_time, noise=0.1): returns actual simulated time (adds multiplicative noise)
- update_model(task, device, actual_time, predicted_time): updates calibration factor (simple EMA)
"""

from typing import Dict
import math
import random


class ExecutionTimePredictor:
    def __init__(self, calibration_init: float = 1.0, ema_alpha: float = 0.05, history_size: int = 1000):
        # multiplicative calibration factor (applied to predicted_time)
        self.calibration_factor = float(calibration_init)
        # EMA alpha for updating calibration factor
        self.ema_alpha = float(ema_alpha)
        # light history for analysis/debug
        self.history = []  # store tuples (task_id, device_id, pred, actual, calib)
        self.history_size = int(history_size)

    def predict_execution_time(self, task, device) -> float:
        """
        Predict execution time (seconds) for `task` on `device` using deterministic model.
        task: object with attributes F, M, P, S
        device: object with attributes F_peak, B_peak, eff_comp, eff_mem, latency
        """
        # effective capabilities
        eff_comp = max(1e-12, device.F_peak * device.eff_comp)
        eff_mem = max(1e-12, device.B_peak * device.eff_mem)

        compute_time = float(task.F) / eff_comp
        memory_time = float(task.M) / eff_mem

        base = max(compute_time, memory_time)

        # parallel factor: P > 1 speeds up (we cap effective parallelism to >=1)
        P_eff = max(1.0, float(task.P))

        # specialization: tasks with high S may run relatively faster on better devices
        # we model it by scaling down base when device looks 'good' for specialization:
        # interpret device 'goodness' as eff_comp normalized to device.F_peak (range 0..1)
        dev_comp_ratio = device.eff_comp  # already 0..1 in our device model
        specialization_gain = 1.0 - (task.S * 0.3 * dev_comp_ratio)  # up to ~30% reduction if S=1 and eff_comp=1
        specialization_gain = max(0.6, specialization_gain)  # bound to avoid too extreme

        pred = (base / P_eff) * specialization_gain + device.latency

        # apply calibration multiplier
        pred *= self.calibration_factor

        # ensure non-zero positive
        pred = max(pred, 1e-6)
        return float(pred)

    def simulate_actual_time(self, predicted_time: float, noise_ratio: float = 0.15) -> float:
        """
        Simulate an actual execution time given a prediction, by multiplicative noise.
        noise_ratio: e.g. 0.15 means uniform noise ~ U(1-0.15, 1+0.15)
        """
        low = max(0.0, 1.0 - float(noise_ratio))
        high = 1.0 + float(noise_ratio)
        factor = random.uniform(low, high)
        actual = float(predicted_time) * factor
        return max(actual, 1e-6)

    def update_model(self, task, device, actual_time: float, predicted_time: float):
        """
        Update calibration factor using a simple EMA towards actual/predicted ratio.
        calibration_factor <- (1-alpha)*calibration_factor + alpha * (actual/pred)
        """
        if predicted_time <= 0:
            return

        ratio = float(actual_time) / float(predicted_time)
        new_calib = (1.0 - self.ema_alpha) * self.calibration_factor + self.ema_alpha * ratio
        self.calibration_factor = float(new_calib)

        # append to history (bounded)
        self.history.append({
            'task_id': getattr(task, 'id', None),
            'device_id': getattr(device, 'id', None),
            'predicted': float(predicted_time),
            'actual': float(actual_time),
            'calibration': float(self.calibration_factor)
        })
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def get_avg_error(self):
        """Return mean absolute relative error from history, if present."""
        if not self.history:
            return 0.0
        errs = [abs((h['actual'] - h['predicted']) / max(h['actual'], 1e-12)) for h in self.history]
        return sum(errs) / len(errs)

    def reset(self):
        self.calibration_factor = 1.0
        self.history.clear()

    def __repr__(self):
        return f"ExecutionTimePredictor(calib={self.calibration_factor:.4f}, history={len(self.history)})"


# quick self-test
if __name__ == "__main__":
    # small ad-hoc test using simple Task/Device from this repo (if available)
    from task import Task
    from device import Device

    pred = ExecutionTimePredictor()
    t = Task(0, F=1e9, M=1e6, P=2.0, S=0.5)
    d = Device("cpu", F_peak=10e9, B_peak=50e9, eff_comp=0.8, eff_mem=0.8, latency=0.001)
    ptime = pred.predict_execution_time(t, d)
    print("predicted:", ptime)
    actual = pred.simulate_actual_time(ptime, noise_ratio=0.2)
    print("actual:", actual)
    pred.update_model(t, d, actual, ptime)
    print("after update:", pred)
