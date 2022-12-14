
from ProjectorEnvironment import ProjectorEnvironment
import torchvision

env = ProjectorEnvironment("sichar.png")
# env.NewImage("balloons.png")
env.NewRandomWall()
env.Snapshot()
env.close()