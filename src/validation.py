# Comments in English only
from typing import List, Dict
import math


def validate_inputs(inp: Dict[str, float]) -> List[str]:
    errors = []

    f_hz = float(inp.get("f_hz", 0.0))
    dt = float(inp.get("dt", 0.0))
    t_end = float(inp.get("t_end", 0.0))
    pre_fault_cycles = float(inp.get("pre_fault_cycles", 0.0))

    S = float(inp.get("S", 0.0))
    Vs = float(inp.get("Vs", 0.0))
    N = float(inp.get("N", 0.0))

    Rw = float(inp.get("Rw", 0.0))
    Rb = float(inp.get("Rb", 0.0))
    Xb = float(inp.get("Xb", 0.0))

    Ip = float(inp.get("Ip", 0.0))
    Off = float(inp.get("Off", 0.0))
    XoverR = float(inp.get("XoverR", 0.0))
    Lamrem = float(inp.get("Lamrem", 0.0))

    if f_hz <= 0.0:
        errors.append("f_hz deve ser > 0.")
    if dt <= 0.0:
        errors.append("dt deve ser > 0.")
    if t_end <= 0.0:
        errors.append("t_end deve ser > 0.")
    if pre_fault_cycles < 0.0:
        errors.append("pre_fault_cycles deve ser >= 0.")

    if Vs <= 0.0:
        errors.append("Vs deve ser > 0.")
    if S <= 1.0:
        errors.append("S deve ser > 1.")
    if N <= 0.0:
        errors.append("N deve ser > 0.")

    if Rw < 0.0 or Rb < 0.0:
        errors.append("Rw e Rb devem ser >= 0.")
    if Xb < 0.0:
        errors.append("Xb deve ser >= 0.")

    if Ip < 0.0:
        errors.append("Ip deve ser >= 0.")

    if Off < -1.0 or Off > 1.0:
        errors.append("Off deve estar no intervalo [-1, 1].")

    if XoverR <= 0.0:
        errors.append("XoverR deve ser > 0.")

    if Lamrem < -1.0 or Lamrem > 1.0:
        errors.append("Lamrem deve estar no intervalo [-1, 1].")

    return errors


def build_warnings(inp: Dict[str, float]) -> List[str]:
    warnings = []

    f_hz = float(inp["f_hz"])
    dt = float(inp["dt"])
    t_end = float(inp["t_end"])
    XoverR = float(inp["XoverR"])
    Off = float(inp["Off"])

    samples_per_cycle = int(round((1.0 / f_hz) / dt))
    if samples_per_cycle < 100:
        warnings.append(f"Samples per cycle = {samples_per_cycle}. Recomenda-se >= 100 (reduza dt).")

    omega = 2.0 * math.pi * f_hz
    tau1 = float(XoverR / omega)
    if t_end < 5.0 * tau1:
        warnings.append(f"t_end={t_end:.4f}s é menor que ~5*tau1={5.0*tau1:.4f}s. Offset DC pode não ter decaído.")

    if abs(Off) > 0.9 and samples_per_cycle < 200:
        warnings.append("Off alto com baixa resolução temporal pode aumentar erro numérico. Considere reduzir dt.")

    return warnings
