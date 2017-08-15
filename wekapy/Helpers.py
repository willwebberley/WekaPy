from wekapy.WekaPyException import WekaPyException
import subprocess
import time


def decode_data(data):
    return data.decode('utf-8').strip()


def run_process(options):
    start_time = time.time()
    process = subprocess.Popen(options, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_output, process_error = map(decode_data, process.communicate())
    if any(word in process_error for word in ["Exception", "Error"]):
        for line in process_error.split("\n"):
            if any(word in line for word in ["Exception", "Error"]):
                raise WekaPyException(line.split(' ', 1)[1])
    end_time = time.time()
    return process_output, end_time - start_time
