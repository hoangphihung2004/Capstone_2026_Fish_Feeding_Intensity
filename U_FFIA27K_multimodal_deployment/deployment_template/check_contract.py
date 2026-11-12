"""
Check that a deployment exposes the same API as Deployment 1 (the dashboard contract).

Usage (on the Jetson, while your deployment is the active one):
    python3 check_contract.py                         # http://127.0.0.1:8012
    python3 check_contract.py --url http://127.0.0.1:8013
    python3 check_contract.py --same-prediction       # also require the same class as Deployment 1

The reference response (contract/reference_step_10.json) was recorded from
Deployment 1 at step 10. Because every deployment reads the same simulation
stream, the audio telemetry at step 10 must be identical; this proves your
container receives exactly the same input window as Deployment 1.
Only the Python standard library is used.
"""

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

CLASS_NAMES = ["none", "strong", "medium", "weak"]
HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "contract", "reference_step_10.json")

results = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not ok else ""))


def fetch(url: str, origin: str = ""):
    req = urllib.request.Request(url, headers={"Origin": origin} if origin else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8")), dict(r.headers)


def kind(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if v is None:
        return "null"
    return type(v).__name__


def compare_structure(ref, got, path=""):
    """Every key/type of the reference must exist in the response (extra keys are allowed)."""
    problems = []
    if isinstance(ref, dict):
        if not isinstance(got, dict):
            return [f"{path or '/'}: expected object, got {kind(got)}"]
        for k, v in ref.items():
            if k not in got:
                problems.append(f"{path}/{k}: missing")
            else:
                problems += compare_structure(v, got[k], f"{path}/{k}")
    elif isinstance(ref, list):
        if not isinstance(got, list):
            return [f"{path}: expected list, got {kind(got)}"]
        if ref and got:
            problems += compare_structure(ref[0], got[0], f"{path}[0]")
    elif kind(ref) != kind(got) and not (ref is None or got is None):
        problems.append(f"{path}: expected {kind(ref)}, got {kind(got)}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8012", help="Base URL of the deployment to check")
    ap.add_argument("--same-prediction", action="store_true", help="Require the same predicted class as Deployment 1")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    ref = json.load(open(REFERENCE, encoding="utf-8"))
    host = base.split("://", 1)[-1].split(":")[0]
    origin = f"http://{host}:8001"

    print(f"Checking {base} against {os.path.relpath(REFERENCE)}\n")

    print("1. GET /api/status")
    try:
        status, headers = fetch(base + "/api/status", origin)
    except (urllib.error.URLError, OSError) as e:
        check(False, "reachable", str(e))
        print("\nRESULT: FAIL (deployment not reachable - is it the active deployment?)")
        sys.exit(1)
    check(status.get("status") == "online", 'status == "online"', str(status.get("status")))
    allow = headers.get("access-control-allow-origin") or headers.get("Access-Control-Allow-Origin")
    check(allow in (origin, "*"), "CORS allows the dashboard origin", f"got {allow!r}")

    print("\n2. GET /api/live/step?step=10  (structure)")
    data, _ = fetch(base + "/api/live/step?step=10")
    problems = compare_structure(ref, data)
    check(not problems, "same fields and types as Deployment 1", "; ".join(problems[:8]))

    print("\n3. Values")
    pred = data.get("predicted_class")
    probs = data.get("probabilities", {})
    check(pred in (0, 1, 2, 3), "predicted_class in 0..3", str(pred))
    if pred in (0, 1, 2, 3):
        check(data.get("class_name") == CLASS_NAMES[pred], "class_name matches predicted_class", str(data.get("class_name")))
        total = sum(float(probs.get(c, 0)) for c in CLASS_NAMES)
        check(abs(total - 100.0) < 0.5, "probabilities sum to 100", f"{total:.2f}")
        best = max(CLASS_NAMES, key=lambda c: float(probs.get(c, -1)))
        check(best == CLASS_NAMES[pred], "argmax(probabilities) == predicted_class", best)
        check(abs(float(data.get("confidence", -1)) - float(probs.get(CLASS_NAMES[pred], -2))) < 0.05,
              "confidence == probability of predicted class")
        relay = data.get("feeder_action", {}).get("relay")
        check(relay == ("ON" if pred in (1, 2) else "OFF"), "feeder_action follows Deployment 1 rule", str(relay))
    lat = data.get("latency", {})
    check(all(isinstance(lat.get(k), (int, float)) and lat.get(k) >= 0 for k in ("audio_ms", "video_ms", "gpu_ms", "total_ms")),
          "latency values present and >= 0", str(lat))
    si = data.get("stream_info", {})
    check(si.get("step") == 10 and si.get("next_step") == 11 and math.isclose(float(si.get("time_sec", -1)), 20.0),
          "stream_info: step 10 -> time_sec 20.0, next_step 11", str(si))
    wf = data.get("audio_telemetry", {}).get("waveform", [])
    check(len(wf) == 600 and all(len(x) == 3 for x in wf), "audio waveform has 600 x [min, max, rms]", f"len={len(wf)}")

    print("\n4. Same input as Deployment 1")
    ra, ga = ref["audio_telemetry"], data.get("audio_telemetry", {})
    check(ga.get("rms_db") == ra["rms_db"] and ga.get("peak_amp") == ra["peak_amp"] and ga.get("waveform") == ra["waveform"],
          "audio window at step 10 is identical", f"rms_db {ga.get('rms_db')} vs {ra['rms_db']}")
    nxt, _ = fetch(base + "/api/live/step?step=11")
    check(nxt.get("stream_info", {}).get("step") == 11, "sequential call with step=11 returns step 11")
    if args.same_prediction:
        check(pred == ref["predicted_class"], "same predicted class as Deployment 1",
              f"{pred} vs {ref['predicted_class']}")
        diff = max(abs(float(probs.get(c, 0)) - float(ref["probabilities"][c])) for c in CLASS_NAMES)
        check(diff < 0.5, "probabilities within 0.5 percentage points of Deployment 1", f"max diff {diff:.3f}")

    passed = sum(results)
    print(f"\nRESULT: {'PASS' if all(results) else 'FAIL'} ({passed}/{len(results)} checks)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
