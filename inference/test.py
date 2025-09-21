import datasets
import json
import os
import yaml


def create_config(
    github_url,
    base_commit,
    problem_id,
    problem_text,
):
    config = {
        'env': {
            'deployment': {
                'name': problem_id,
                'type': 'conda',
                'python': "3.11",
                'remove_env_on_stop': True,
                'clear_conda_workspace': True
            },
            'repo': {
                'type': 'github',
                'github_url': github_url,
                'base_commit': base_commit
            }
        },
        'problem_statement': {
            'id': problem_id,
            'type': 'text',
            'text': problem_text
        }
    }
    return config


data = datasets.load_dataset("SumanthRH/SWE-Gym-Subset", split="train")
dir = f"/tmp/SWE-agent/data/evals"


config_list = []

for i in range(71, 80):
    instance_id = data[i]["instance_id"]
    with open(os.path.join(dir, f"{instance_id}.test.patch"), 'w') as f:
        f.write(data[i]["test_patch"])
    with open(os.path.join(dir, f"{instance_id}.sh"), 'w') as f:
        f.write(data[i]["eval_script"])
    with open(os.path.join(dir, f"{instance_id}_fail.json"), 'w') as f:
        f.write(json.dumps(data[i]["FAIL_TO_PASS"]))
    with open(os.path.join(dir, f"{instance_id}_pass.json"), 'w') as f:
        f.write(json.dumps(data[i]["PASS_TO_PASS"]))

    config = create_config(
        github_url=f'https://github.com/{data[i]["repo"]}',
        base_commit=data[i]["base_commit"],
        problem_id=instance_id,
        problem_text=data[i]["problem_statement"],
    )
    config_list.append(config)


with open("/tmp/SWE-agent/config/all_instances.yaml", 'w') as file:
    yaml.dump(config_list, file, 
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
                sort_keys=False)
