import urllib.request
import json
import time
import sys

campaign_url = 'http://127.0.0.1:8300/api/campaign'
state_url = 'http://127.0.0.1:8300/api/state'

def post_json(url, data_dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(data_dict).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode('utf-8'))

def get_state():
    with urllib.request.urlopen(state_url) as r:
        return json.loads(r.read().decode('utf-8'))

# 1. First POST {"mode":"rehearsal"}
print("Triggering rehearsal...")
response = post_json(campaign_url, {"mode": "rehearsal"})
print("Response:", response)

# 2. Monitor until idle (max 5 min = 300s, poll >= 3s)
print("Monitoring rehearsal...")
t0 = time.time()
rehearsal_state = None
while True:
    try:
        rehearsal_state = get_state()
        state_str = rehearsal_state.get('state')
        progress = rehearsal_state.get('progress', {})
        print(f"[{int(time.time() - t0)}s] State: {state_str}, Progress: {progress}")
        if state_str == 'idle':
            print("Reached idle state.")
            break
    except Exception as e:
        print(f"Error polling: {e}")
    
    if time.time() - t0 > 300:
        print("Timeout of 5 minutes reached in rehearsal.")
        break
    time.sleep(4)

# Print final state properties
print("\n--- REHEARSAL SUMMARY ---")
print(f"rehearsal_passed: {rehearsal_state.get('rehearsal_passed')}")
print(f"result: {rehearsal_state.get('result')}")
print(f"collision: {rehearsal_state.get('collision')}")
print(f"latest telemetry sample: {rehearsal_state.get('sample')}")
print("-------------------------\n")

# Check if rehearsal passed
rehearsal_passed = rehearsal_state.get('rehearsal_passed')
result = rehearsal_state.get('result') or {}
result_status = result.get('status')

# If and only if rehearsal_passed is true, result complete and not aborted,
should_proceed = False
if rehearsal_passed is True and result_status == 'complete' and result.get('comment') != 'aborted':
    should_proceed = True
elif rehearsal_passed is True and result_status == 'completed' and result.get('comment') != 'aborted':
    # Accept complete or completed or whatever status denotes success
    should_proceed = True

# Wait, let's also look at the direct description of the request:
# "If and only if rehearsal_passed is true, result complete and not aborted, then POST exactly ..."
# Let's print the actual values to be completely sure.
if should_proceed:
    print("Conditions met. Posting optimal_excitation...")
    optimal_payload = {
        "mode": "optimal_excitation",
        "options": {
            "optimal_training_trajectories": 12,
            "optimal_validation_trajectories": 3,
            "optimal_friction_postures": 3,
            "optimal_friction_repeats": 2,
            "fourier_base_frequency_hz": 0.09,
            "fourier_duration_s": 30,
            "reuse_friction": True
        }
    }
    opt_resp = post_json(campaign_url, optimal_payload)
    print("Optimal excitation response:", opt_resp)
    
    # Poll recovery state for up to 60s (>=2s interval),
    # stopping when reuse note appears and phase is starting/C_inertia, or idle/failed.
    print("Polling recovery state...")
    t1 = time.time()
    while True:
        try:
            curr_state = get_state()
            state_str = curr_state.get('state')
            progress = curr_state.get('progress') or {}
            phase = progress.get('phase')
            notes = curr_state.get('notes') or []
            
            # check if reuse note appears: 'reuse' in note
            reuse_note_appears = any('reuse' in n.lower() for n in notes)
            
            print(f"[{int(time.time() - t1)}s] State: {state_str}, Phase: {phase}, Reuse note: {reuse_note_appears}")
            
            phase_lower = str(phase).lower()
            # stopping when reuse note appears and phase is starting/C_inertia, or idle/failed
            is_phase_starting_or_cinertia = 'start' in phase_lower or 'c_inertia' in phase_lower or 'inertia' in phase_lower
            
            if (reuse_note_appears and is_phase_starting_or_cinertia) or state_str in ['idle', 'failed']:
                print("Stopping condition met!")
                print(f"Final notes: {notes}")
                print(f"Final progress: {progress}")
                print(f"Final state: {curr_state}")
                break
        except Exception as e:
            print(f"Error polling recovery state: {e}")
            
        if time.time() - t1 > 60:
            print("60s polling timeout reached.")
            break
        time.sleep(2)
else:
    print("Conditions not met to proceed to optimal excitation.")
