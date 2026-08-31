import urllib.request
import json
import time
import sys

url = "http://127.0.0.1:8300/api/state"

def get_state():
    try:
        req = urllib.request.urlopen(url, timeout=5)
        return json.loads(req.read().decode())
    except Exception as e:
        print(f"Error fetching state: {e}", file=sys.stderr)
        return None

initial_state = get_state()
if not initial_state:
    print("Failed to fetch initial state", file=sys.stderr)
    sys.exit(1)

initial_notes = set(initial_state.get('notes', []))
joint_names = initial_state.get('joint_names', [])
num_joints = len(joint_names)

max_abs_current = [0.0] * num_joints
max_temp = [0.0] * num_joints

initial_phase = initial_state.get('progress', {}).get('phase')
print(f"Starting monitor. Initial phase: {initial_phase}, pass: {initial_state.get('progress', {}).get('pass')}")

start_time = time.time()
limit_seconds = 1200 # 20 minutes
poll_interval = 15

last_good_state = initial_state

def update_maxima(state):
    sample = state.get('sample', {})
    currents = sample.get('current_a', [])
    temps = sample.get('temperature_c', [])
    for i in range(min(num_joints, len(currents))):
        c_abs = abs(currents[i])
        if c_abs > max_abs_current[i]:
            max_abs_current[i] = c_abs
    for i in range(min(num_joints, len(temps))):
        t = temps[i]
        if t > max_temp[i]:
            max_temp[i] = t

update_maxima(initial_state)

try:
    while True:
        elapsed = time.time() - start_time
        if elapsed >= limit_seconds:
            print("Time limit of 20 minutes reached.")
            break
            
        time.sleep(poll_interval)
        
        state = get_state()
        if not state:
            continue
            
        last_good_state = state
        update_maxima(state)
        
        run_state = state.get('state')
        progress = state.get('progress', {})
        phase = progress.get('phase')
        passes = progress.get('pass')
        
        sample = state.get('sample', {})
        temps = sample.get('temperature_c', [])
        
        print(f"[{int(elapsed)}s] State: {run_state}, Phase: {phase}, Pass: {passes}, Max Temp: {max(temps) if temps else 'N/A'}")
        sys.stdout.flush()
        
        if run_state != 'running':
            print(f"State is no longer running. Current state: {run_state}")
            break
            
        if phase != initial_phase:
            print(f"Phase changed from {initial_phase} to {phase}")
            break
            
        if initial_phase == 'B_friction' and passes is not None and passes >= 500:
            print(f"progress.pass reached >= 500 (value: {passes}) in B_friction phase.")
            break
            
        if temps and any(t >= 40.0 for t in temps):
            print(f"Temperature limit reached. A joint temperature is >= 40C: {temps}")
            break

except KeyboardInterrupt:
    print("Monitor interrupted by user.")

# Collect results
print("\n=== MONITOR FINAL RESULT ===")
res = {}
res['state'] = last_good_state.get('state')
res['activity'] = last_good_state.get('activity')
res['progress'] = last_good_state.get('progress')

maxima = {}
for i, name in enumerate(joint_names):
    maxima[name] = {
        'max_abs_current_a': max_abs_current[i],
        'max_temperature_c': max_temp[i]
    }
res['maxima'] = maxima

sample = last_good_state.get('sample', {})
res['latest'] = {
    'position_deg': sample.get('position_deg'),
    'current_a': sample.get('current_a'),
    'temperature_c': sample.get('temperature_c'),
    'enabled': sample.get('enabled'),
    'fault_code': sample.get('fault_code'),
    'collision': last_good_state.get('collision')
}

final_notes = last_good_state.get('notes', [])
new_notes = [n for n in final_notes if n not in initial_notes]
res['new_notes'] = new_notes

print(json.dumps(res, indent=2))
