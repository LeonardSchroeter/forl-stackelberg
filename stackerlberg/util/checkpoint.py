import os

def _parse_name(filename):
    parts = filename.split(".")[0].split("_")
    kind, steps = parts[0], parts[1]
    steps = int(steps)
    return {
        "kind": kind,
        "steps": steps
    }

def _latest_n_checkpoint_steps(base_path, n=5):
    steps = set(map(lambda x: _parse_name(x)['steps'], os.listdir(base_path)))
    latest_steps = sorted(steps)
    latest_n = latest_steps[-n:]
    return latest_n


def _latest_step(base_path):
    return _latest_n_checkpoint_steps(base_path, n=1)[-1]