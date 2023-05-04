# Dissertation

## Robot-Interception-with-vision-based-deep-reinforcement-learning
Core codes of the project, not complete.
### Running code
1. Install dependencies in `requirements.txt` by
```shell
pip install -r requirements.txt
```
2. Download the files in `./Sim-Env/` and then put them in `./gym/envs/final/`.
3. Create a `__init__.py` file inside `./gym/envs/final/` and paste the following 
```python
from gym.envs.final.DiscreteRobotInterception import DiscreteRobotMovingEnv
from gym.envs.final.ContinuousRobotInterception import TwoWheelRobotContinuousMovingEnv
```
4. Open `./gym/envs/__init__.py` and add the following code at the end of the file
```python
register(
    id="RobotInterception-v1",
    entry_point="gym.envs.final:DiscreteRobotMovingEnv",
    max_episode_steps=10000,
)

register(
    id="RobotInterception-v4",
    entry_point="gym.envs.final:TwoWheelRobotContinuousMovingEnv",
    max_episode_steps=10000,
)
```
5. Then, you can run `main.py` in each folder.

### References
- [YOLO](https://github.com/ultralytics/yolov5)
- [PPO](https://github.com/nikhilbarhate99/PPO-PyTorch)
- [PPO-GAE](https://github.com/Lizhi-sjtu/DRL-code-pytorch)

