import datetime
import os
from mlagents_envs.environment import UnityEnvironment, ActionTuple
import numpy as np
import json
import platform

class ProjectorEnvironment:
  def __init__(self, imageFilePath):
    if (platform.system() == "Linux"):
      self.folder = "linux_exe"
      self.file_name = "ProjectorEnvironment.x86_64"
    if (platform.system() == "Windows"):
      self.folder = "windows_exe"
      self.file_name = "ProjectorEnvironment.exe"
    if (platform.system() == "Darwin"):
      # throw an unsupported exception
      raise Exception("MacOS is not supported")
    
    self.flatwall = {
      "bottomLeft": 0,
      "bottomRight": 0,
      "bottomMiddle": 0,
      "middleLeft": 0,
      "middleRight": 0,
      "middleMiddle": 0,
      "topLeft": 0,
      "topRight": 0,
      "topMiddle": 0
    }
    self.SetWall(self.flatwall)
    self.env = UnityEnvironment(file_name=f"{self.folder}/{self.file_name}")
    self.env.reset()
    self.behavior_name = list(self.env.behavior_specs)[0]
    self.currFolder = ""
    self.wall = self.flatwall
    self.NewImage(imageFilePath, new_class=True)
    
  def about(self):
    # get groups
    spec = self.env.behavior_specs[self.behavior_name]

    if spec.action_spec.continuous_size > 0:
      print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
      print(f"There are {spec.action_spec.discrete_size} discrete actions")

    # For discrete actions only : How many different options does each action has ?
    if spec.action_spec.discrete_size > 0:
      for action, branch_size in enumerate(spec.action_spec.discrete_branches):
        print(f"Action number {action} has {branch_size} different options")

  def Snapshot(self, filename=None):
    if (filename is None):
      filename = str(datetime.datetime.utcnow().timestamp()) + ".png"
    actions = np.array([[1,0,0]])
    action_tuple = ActionTuple()
    action_tuple.add_discrete(actions)
    self.env.set_actions(self.behavior_name, action_tuple)
    self.env.step()
    # copy snapshot to folder with filename
    with open(f"{self.currFolder}/{filename}", "wb") as f:
      with open(f"{self.folder}/snapshot.png", "rb") as f2:
        f.write(f2.read())
    
  def NewImage(self, filepath, new_class = False):
    # overwrite currentImage.png
    with open(f"{self.folder}/currentImage.png", "wb") as f:
      with open(filepath, "rb") as f2:
        f.write(f2.read())

    actions = np.array([[0,1,0]])
    action_tuple = ActionTuple()
    action_tuple.add_discrete(actions)
    self.env.set_actions(self.behavior_name, action_tuple)
    self.env.step()
    
    if (new_class):
      self.NewFolder(filepath)
    
  def NewFolder(self, filepath):
    self.currFolder = filepath.split('.')[0]
    if not os.path.exists(self.currFolder):
      # create a folder for the image
      os.mkdir(self.currFolder)
    
    if not os.path.exists(f"{self.currFolder}/label.png"):
      previousWall = self.wall
      # take the flat wall snapshot
      self.FlattenWall()
      self.Snapshot("label.png")
      # return to previous wall
      self.SetWall(previousWall)

  def SetWall(self,wall):
    # wall to json
    self.wall = wall
    wallJson = json.dumps(wall)
    # save to file
    with open(f"{self.folder}/wallPositions.json", "w") as f:
      f.write(wallJson)
    actions = np.array([[0,0,1]])
    action_tuple = ActionTuple()
    action_tuple.add_discrete(actions)
    # if self has attribute env
    if hasattr(self, 'env'):
      self.env.set_actions(self.behavior_name, action_tuple)
      self.env.step()

  def newRandomWall(self):
    nums = [np.random.rand() for x in range(9)]
    wall = {
      "bottomLeft": nums[0],
      "bottomRight": nums[1],
      "bottomMiddle": nums[2],
      "middleLeft": nums[3],
      "middleRight": nums[4],
      "middleMiddle": nums[5],
      "topLeft": nums[6],
      "topRight": nums[7],
      "topMiddle": nums[8]
    }
    self.SetWall(wall)
    
  def FlattenWall(self):
    self.SetWall(self.flatwall)
    
  def close(self):
    self.env.close()