import datasets
import json

data = datasets.load_dataset("SumanthRH/SWE-Gym-Subset", split="train")[19]
with open("/tmp/SWE-agent/trajectories/root/default_bash_only__openai--Qwen--Qwen3-8B__t-0.70__p-0.95__c-0.00___instances/python__mypy-14064/python__mypy-14064.patch", 'w') as f:
    f.write(data["patch"])