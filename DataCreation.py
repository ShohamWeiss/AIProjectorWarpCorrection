
from ProjectorEnvironment import ProjectorEnvironment

env = ProjectorEnvironment("sichar.png")
env.newImage("balloons.png")
env.newRandomWall()
env.snapshot()
env.close()