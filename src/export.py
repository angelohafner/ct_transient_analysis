# Comments in English only
from typing import Dict, Any
import io
import csv
import json


def results_to_csv_bytes(res: Dict[str, Any]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["t", "lambda", "is", "ie", "i2", "is_rms", "i2_rms"])

    t = res["t"]
    lam = res["lam"]
    is_arr = res["is"]
    ie_arr = res["ie"]
    i2_arr = res["i2"]
    is_rms = res["is_rms"]
    i2_rms = res["i2_rms"]

    n = int(len(t))
    i = 0
    while i < n:
        writer.writerow([
            float(t[i]),
            float(lam[i]),
            float(is_arr[i]),
            float(ie_arr[i]),
            float(i2_arr[i]),
            float(is_rms[i]),
            float(i2_rms[i])
        ])
        i = i + 1

    return buf.getvalue().encode("utf-8")


def results_to_json_bytes(inp: Dict[str, float], derived: Dict[str, float]) -> bytes:
    payload = {}
    payload["inp"] = dict(inp)
    payload["derived"] = dict(derived)
    return json.dumps(payload, indent=2).encode("utf-8")
