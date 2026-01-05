# Comments in English only
from typing import List


PRESETS = {
    "Sem saturação": {
        "description": "Burden baixo e corrente moderada. Espera-se i2(t) ~ is(t), ie pequeno e pouca distorção.",
        "inp": {
            "f_hz": 60.0,
            "t_end": 0.45,
            "pre_fault_cycles": 1.0,
            "dt": 1.0 / 12000.0,
            "S": 22.0,
            "Vs": 400.0,
            "N": 240.0,
            "Rw": 0.2,
            "Rb": 0.5,
            "Xb": 0.2,
            "Ip": 4000.0,
            "Off": 0.0,
            "XoverR": 12.0,
            "Lamrem": 0.0
        }
    },
    "Saturação forte": {
        "description": "Burden alto e corrente elevada. Espera-se distorção forte em i2(t) e ie elevado.",
        "inp": {
            "f_hz": 60.0,
            "t_end": 0.45,
            "pre_fault_cycles": 1.0,
            "dt": 1.0 / 24000.0,
            "S": 22.0,
            "Vs": 400.0,
            "N": 240.0,
            "Rw": 0.2,
            "Rb": 6.0,
            "Xb": 4.0,
            "Ip": 15000.0,
            "Off": 0.3,
            "XoverR": 12.0,
            "Lamrem": 0.0
        }
    },
    "Offset DC severo": {
        "description": "Off alto e X/R típico. Espera-se saturação mais intensa no primeiro semiciclo.",
        "inp": {
            "f_hz": 60.0,
            "t_end": 0.45,
            "pre_fault_cycles": 1.0,
            "dt": 1.0 / 24000.0,
            "S": 22.0,
            "Vs": 400.0,
            "N": 240.0,
            "Rw": 0.2,
            "Rb": 4.0,
            "Xb": 2.0,
            "Ip": 12000.0,
            "Off": 0.95,
            "XoverR": 12.0,
            "Lamrem": 0.0
        }
    },
    "Com remanência": {
        "description": "Aplica remanência em λ. Espera-se deslocamento e assimetria adicional no fluxo e correntes.",
        "inp": {
            "f_hz": 60.0,
            "t_end": 0.45,
            "pre_fault_cycles": 1.0,
            "dt": 1.0 / 24000.0,
            "S": 22.0,
            "Vs": 400.0,
            "N": 240.0,
            "Rw": 0.2,
            "Rb": 4.0,
            "Xb": 2.0,
            "Ip": 12000.0,
            "Off": 0.3,
            "XoverR": 12.0,
            "Lamrem": 0.3
        }
    }
}


def preset_names() -> List[str]:
    return list(PRESETS.keys())
