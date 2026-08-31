import urllib.request
import json
import time

url_campaign = "http://127.0.0.1:8300/api/campaign"
url_state = "http://127.0.0.1:8300/api/state"

# 1. POST {"mode":"rehearsal"}
data = json.dumps({"mode": "rehearsal"}).encode('utf-8')
req = urllib.request.Request(url_campaign, data=data, headers={'Content-Type': 'application/json'}, method='POST')

print("Sending POST request to start rehearsal...")
try:
    with urllib.request.urlopen(req) as response:
        post_resp_code = response.getcode()
        post_resp_body = response.read().decode('utf-8')
        print(f"POST Response Code: {post_resp_code}")
        print(f"POST Response Body:\n{post_resp_body}")
        post_data = json.loads(post_resp_body)
except Exception as e:
    print(f"POST failed: {e}")
    post_data = None

print("\nStarting monitoring...")
start_time = time.time()
phase_counts = {}
last_phase = None
started = False

# We'll monitor for up to 10 minutes (600 seconds)
# Poll every 5 seconds.
while True:
    time.sleep(5)
    elapsed = time.time() - start_time
    if elapsed > 600:
        print("Timeout reached (10 minutes). Stopping.")
        break

    try:
        req_state = urllib.request.Request(url_state)
        with urllib.request.urlopen(req_state) as response:
            state_data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching state at elapsed {elapsed:.1f}s: {e}")
        continue

    state_str = state_data.get("state", "")
    progress = state_data.get("progress", {})
    phase = progress.get("phase", "unknown")
    
    # Track phase observation counts
    phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    if phase != "idle" and phase != "unknown":
        started = True

    print(f"Elapsed: {elapsed:.1f}s | State: {state_str} | Phase: {phase}")

    # Check termination condition:
    # If we have started (or even if we haven't after some time) and the state is 'idle'
    if started and state_str == "idle":
        print("State returned to 'idle'. Rehearsal campaign completed/terminated.")
        # Print final details
        print("\n=== FINAL RESULTS ===")
        print(f"Final state: {state_str}")
        print(f"Rehearsal passed: {state_data.get('rehearsal_passed')}")
        print(f"Progress: {json.dumps(progress, indent=2)}")
        print(f"Result: {json.dumps(state_data.get('result'), indent=2)}")
        print(f"Phase Observation Counts: {phase_counts}")
        
        # Validation RMS summary validation can be checked from result or notes
        result = state_data.get('result') or {}
        print(f"Result summary keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        print("\n=== Latest Notes ===")
        notes = state_data.get('notes', [])
        for note in notes[-15:]:
            print(note)
        break

