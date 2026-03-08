import json
import os

class PerformanceMetrics:
    def __init__(self):
        self.results = []

    def log_result(self, algo_name, instance_name, cost, time_taken, parameters=None):
        result = {
            "algorithm": algo_name,
            "instance": instance_name,
            "cost": int(cost),
            "time_seconds": float(time_taken),
            "parameters": parameters or {}
        }
        self.results.append(result)
        return result

    def save_to_json(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=4)
