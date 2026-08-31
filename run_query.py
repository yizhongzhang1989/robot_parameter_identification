import urllib.request
import json
import time

def get_state():
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8301/api/state", timeout=5)
        return json.loads(req.read().decode())
    except Exception as e:
        print(f"Error getting state: {e}")
        return None

def post_json(url, data):
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error posting to {url}: {e}")
        return None

print("=== PART 1: Polling GET /api/state ===")
start_time = time.time()
state = None
while time.time() - start_time < 30:
    state = get_state()
    if state and state.get("have_model") and state.get("ok"):
        sample = state.get("sample")
        if sample:
            print("Model and sample are available!")
            break
    time.time()
    time.sleep(1)

if not state:
    print("Could not retrieve state. Stopping.")
    exit(1)

# Verify step 1 conditions:
# idle, all enabled, faults zero, temps<40, collision clear, current_guard true.
sample = state.get("sample") or {}
state_name = state.get("state")
all_enabled = all(state.get("sample", {}).get("enabled", [])) if state.get("sample") else False
faults_zero = all(f == 0 for f in state.get("sample", {}).get("fault_code", [])) if state.get("sample") else False
temps = state.get("sample", {}).get("temperature_c", []) if state.get("sample") else []
temps_ok = all(t < 40 for t in temps) if temps else False
collision_clear = state.get("collision", {}).get("clear") is True
current_guard = state.get("current_guard") is True

print("State details:")
print(f" - State: {state_name} (expected: idle)")
print(f" - All enabled: {all_enabled} (expected: True)")
print(f" - Faults zero: {faults_zero} (expected: True)")
print(f" - Temps < 40: {temps_ok} (values: {temps}, expected: True)")
print(f" - Collision clear: {collision_clear} (expected: True)")
print(f" - Current guard: {current_guard} (expected: True)")

pre_home_safety = {
    "state": state_name,
    "all_enabled": all_enabled,
    "faults_zero": faults_zero,
    "temps": temps,
    "collision_clear": collision_clear,
    "current_guard": current_guard
}

if not (state_name == "idle" and all_enabled and faults_zero and temps_ok and collision_clear and current_guard):
    print("Pre-requisite check failed inside poll, stopping as requested.")
    print(f"PRE_HOME_SAFETY={json.dumps(pre_home_safety)}")
    exit(0)

print("PART 1 PASSED! Continuing to Part 2...")
