# Comments in English only
import math
from typing import Dict, Tuple, Any

import numpy as np


class CTSatModel:
    def __init__(self, inp: Dict[str, float], derived: Dict[str, float]):
        self.inp = dict(inp)
        self.der = dict(derived)

    @staticmethod
    def compute_rp_numeric(S: float, n_points: int = 200000) -> float:
        theta = np.linspace(0.0, 2.0 * math.pi, int(n_points))
        y = np.sin(theta)
        y = np.power(y, 2.0 * float(S))
        integral = np.trapezoid(y, theta)
        irms_sq = (1.0 / (2.0 * math.pi)) * float(integral)
        rp = math.sqrt(irms_sq)
        return float(rp)

    @staticmethod
    def compute_derived(inp: Dict[str, float], rp: float) -> Dict[str, float]:
        S = float(inp["S"])
        Vs = float(inp["Vs"])
        Rw = float(inp["Rw"])
        Rb = float(inp["Rb"])
        Xb = float(inp["Xb"])
        XoverR = float(inp["XoverR"])
        f_hz = float(inp["f_hz"])

        omega = 2.0 * math.pi * f_hz
        Rt = float(Rw + Rb)
        Tau1 = float(XoverR / omega)
        Lb = float(Xb / omega)

        A_num = 10.0 * (omega ** S)
        A_den = ((Vs * math.sqrt(2.0)) ** S) * float(rp)
        A = float(A_num / A_den)

        Lamsat = float(1.41428 * Vs / omega)

        out = {}
        out["omega"] = float(omega)
        out["Rt"] = float(Rt)
        out["Tau1"] = float(Tau1)
        out["Lb"] = float(Lb)
        out["RP"] = float(rp)
        out["A"] = float(A)
        out["Lamsat"] = float(Lamsat)
        return out

    @staticmethod
    def sliding_rms(x: np.ndarray, window_samples: int) -> np.ndarray:
        if int(window_samples) < 1:
            raise ValueError("window_samples must be >= 1")

        x2 = x * x
        w = np.ones(int(window_samples), dtype=float)

        ma_valid = np.convolve(x2, w, mode="valid") / float(window_samples)
        rms_valid = np.sqrt(ma_valid)

        if int(rms_valid.size) == 0:
            return np.zeros_like(x, dtype=float)

        pad_len = int(window_samples - 1)
        pad = np.full(pad_len, float(rms_valid[0]), dtype=float)
        rms = np.concatenate([pad, rms_valid])
        return rms

    def is_and_disdt(self, t: float) -> Tuple[float, float]:
        Ip = float(self.inp["Ip"])
        N = float(self.inp["N"])
        Off = float(self.inp["Off"])

        omega = float(self.der["omega"])
        Tau1 = float(self.der["Tau1"])

        if float(t) < 0.0:
            return 0.0, 0.0

        off_clip = float(min(1.0, max(-1.0, Off)))
        phi = float(math.acos(off_clip))

        term_dc = Off * math.exp(-t / Tau1)
        term_ac = math.cos((omega * t) - phi)
        is_val = math.sqrt(2.0) * Ip / N * (term_dc - term_ac)

        term_dc_d = (-Off / Tau1) * math.exp(-t / Tau1)
        term_ac_d = omega * math.sin((omega * t) - phi)
        disdt_val = math.sqrt(2.0) * Ip / N * (term_dc_d + term_ac_d)

        return float(is_val), float(disdt_val)

    def ie_of_lambda(self, lam: float) -> float:
        A = float(self.der["A"])
        S = float(self.inp["S"])

        if lam > 0.0:
            sgn = 1.0
        elif lam < 0.0:
            sgn = -1.0
        else:
            sgn = 0.0

        return float(A * sgn * (abs(lam) ** S))

    def die_dlambda(self, lam: float) -> float:
        A = float(self.der["A"])
        S = float(self.inp["S"])
        return float(A * S * (abs(lam) ** (S - 1.0)))

    def rhs_lambda(self, t: float, lam: float) -> float:
        Rt = float(self.der["Rt"])
        Lb = float(self.der["Lb"])

        is_val, disdt_val = self.is_and_disdt(t=float(t))
        ie_val = self.ie_of_lambda(lam=float(lam))

        denom = 1.0 + (Lb * self.die_dlambda(lam=float(lam)))
        rhs = (-Rt * ie_val) + (Rt * is_val) + (Lb * disdt_val)

        return float(rhs / denom)

    def integrate_rk4(self, t: np.ndarray, lam0: float) -> np.ndarray:
        lam = np.zeros_like(t, dtype=float)
        lam[0] = float(lam0)

        k = 0
        while k < int(t.size - 1):
            dt = float(t[k + 1] - t[k])
            tk = float(t[k])
            yk = float(lam[k])

            k1 = self.rhs_lambda(t=tk, lam=yk)
            k2 = self.rhs_lambda(t=tk + 0.5 * dt, lam=yk + 0.5 * dt * k1)
            k3 = self.rhs_lambda(t=tk + 0.5 * dt, lam=yk + 0.5 * dt * k2)
            k4 = self.rhs_lambda(t=tk + dt, lam=yk + dt * k3)

            lam[k + 1] = yk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            k = k + 1

        return lam

    def integrate_scipy(self, t: np.ndarray, lam0: float, rtol: float, atol: float) -> np.ndarray:
        try:
            from scipy.integrate import solve_ivp
        except Exception as e:
            raise RuntimeError("SciPy is not available. Install scipy or use RK4.") from e

        def ode(t_scalar, y_vec):
            lam_val = float(y_vec[0])
            dlam_dt = self.rhs_lambda(t=float(t_scalar), lam=float(lam_val))
            return [float(dlam_dt)]

        sol = solve_ivp(
            fun=ode,
            t_span=(float(t[0]), float(t[-1])),
            y0=[float(lam0)],
            t_eval=t,
            method="RK45",
            rtol=float(rtol),
            atol=float(atol)
        )

        if bool(sol.success) is False:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        return sol.y[0].astype(float)

    def simulate(self, integrator: str, rtol: float = 1.0e-6, atol: float = 1.0e-9) -> Dict[str, Any]:
        f_hz = float(self.inp["f_hz"])
        t_end = float(self.inp["t_end"])
        pre_fault_cycles = float(self.inp["pre_fault_cycles"])
        dt = float(self.inp["dt"])

        t_pre = float(pre_fault_cycles / f_hz)
        t0 = 0.0 - t_pre

        n_steps = int(math.floor((t_end - t0) / dt)) + 1
        t = t0 + dt * np.arange(n_steps, dtype=float)

        lam0 = float(self.der["Lamsat"] * float(self.inp["Lamrem"]) + 0.0)

        if str(integrator).startswith("SciPy"):
            lam = self.integrate_scipy(t=t, lam0=lam0, rtol=float(rtol), atol=float(atol))
        else:
            lam = self.integrate_rk4(t=t, lam0=lam0)

        is_arr = np.zeros_like(t)
        disdt_arr = np.zeros_like(t)
        ie_arr = np.zeros_like(t)
        i2_arr = np.zeros_like(t)

        k = 0
        while k < int(t.size):
            is_val, disdt_val = self.is_and_disdt(t=float(t[k]))
            ie_val = self.ie_of_lambda(lam=float(lam[k]))

            is_arr[k] = float(is_val)
            disdt_arr[k] = float(disdt_val)
            ie_arr[k] = float(ie_val)
            i2_arr[k] = float(is_val - ie_val)

            k = k + 1

        samples_per_cycle = int(round((1.0 / f_hz) / dt))
        is_rms = self.sliding_rms(is_arr, samples_per_cycle)
        i2_rms = self.sliding_rms(i2_arr, samples_per_cycle)

        out = {}
        out["t"] = t
        out["lam"] = lam
        out["is"] = is_arr
        out["ie"] = ie_arr
        out["i2"] = i2_arr
        out["disdt"] = disdt_arr
        out["is_rms"] = is_rms
        out["i2_rms"] = i2_rms
        out["samples_per_cycle"] = int(samples_per_cycle)
        return out
