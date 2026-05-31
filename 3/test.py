import torch as th
import numpy as np
import gymnasium as gym

from pathlib import Path
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

# Model citamo pored ove skripte (u folderu 3).
BASE_DIR = Path(__file__).resolve().parent

if th.cuda.is_available():
    device = "cuda"
elif th.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

LOCAL = True

#################### TO-DO ####################


class SimpleNet(th.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = th.nn.Linear(108, 64, dtype=th.float64)
        self.fc2 = th.nn.Linear(64, 64, dtype=th.float64)
        self.out = th.nn.Linear(64, 7, dtype=th.float64)

    def forward(self, x):
        x = th.nn.functional.relu(self.fc1(x))
        x = th.nn.functional.relu(self.fc2(x))
        return self.out(x)

model = SimpleNet().to(device)
model.load_state_dict(th.load(BASE_DIR / 'model.pth', map_location=device))
model.eval()


###############################################


env = gym.make(
    "MiniGrid-Empty-Random-6x6-v0",
    render_mode="human" if LOCAL else "rgb_array",
    highlight=False,
    screen_size=640
)

env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

rewards = []

for episode in range(10):
    if model is None:
        break

    obs, _ = env.reset()
    step = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and step < 30:
        if LOCAL:
            env.render()
        with th.no_grad():
            obs = th.tensor(obs, dtype=th.float64, device=device).reshape(-1, 108)  #! promeniti po potrebi
            action = model(obs)
        obs, reward, terminated, truncated, _ = env.step(th.argmax(action).item())
        step += 1
    
    print(f"{episode=} {reward=}")
    rewards.append(reward)

env.close()
print(f"mean reward: {np.mean(rewards)}")
