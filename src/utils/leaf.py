import json
from tqdm import tqdm

def load_leaf(paths):
    users = []
    num_samples = []
    user_data = {}
    for path in tqdm(paths, total=len(paths)):
        with open(path) as f:
            data = json.load(f)
        users += data['users']
        num_samples += data['num_samples']
        user_data.update(data['user_data'])
    return users, num_samples, user_data