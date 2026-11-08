"""Collect local runtime memory statistics without recording instance identity."""
import json
import selectors
import subprocess
import time


def capture(output,timeout=6):
    process=subprocess.Popen(['/opt/aws/neuron/bin/neuron-monitor'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    deadline=time.monotonic()+timeout
    samples=[]
    try:
        while time.monotonic()<deadline:
            if not selector.select(max(0,deadline-time.monotonic())):break
            line=process.stdout.readline()
            if not line:break
            data=json.loads(line)
            samples.append({'neuron_runtime_data':data.get('neuron_runtime_data',[]),
                            'system_memory':data.get('system_data',{}).get('memory_info',{}),
                            'neuron_hardware_info':data.get('neuron_hardware_info',{})})
            if len(samples)>=2:break
    finally:
        selector.close();process.terminate()
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    output.write_text(json.dumps({'samples':samples,'note':'Idle after loaded-model quality checks; not a high-water memory measurement.'},indent=2))
    return len(samples)
