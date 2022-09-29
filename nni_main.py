"""
NNI hyperparameter optimization example.

Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

from pathlib import Path
import signal

from nni.experiment import Experiment

# Define search space
search_space = {
    'bs': {'_type': 'choice', '_value': [16, 32, 64, 128]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'alpha': {'_type': 'loguniform', '_value': [0.0001, 1]},
    'T': {'_type': 'uniform', '_value': [3, 1000]}
}


# Configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python3 nni_train_kd.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.max_trial_number = 20
experiment.config.trial_concurrency = 1

# Run it!
experiment.run(port=8080, wait_completion=False)

# windows does not implement signal.pause
signal.pause()
