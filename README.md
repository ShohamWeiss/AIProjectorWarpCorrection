![Alt text](https://github.com/ShohamWeiss/AIProjectorWarpCorrection/blob/0f04af636beccb4e8029a9782ae1a5e35e1432fe/Screenshot%202022-12-13%20191906.jpg)
# AI Projector Warp Correction Environment
Projector Python Environment to create data and train deep learning models. The environment consists of a projector, a wall and a camera. The wall is made out of a 3x3 grid of blocks that can be aranged via the sdk to any orientation to create a non-flat surface to project onto. The sdk also allows for setting an image to project onto the wall and to take a snapshot of the projected image.

## What is it?
A unity environment that's wrapped by a python library to make a data creation sdk for training machine learning models.

## How to use
```python
from ProjectorEnvironment import ProjectorEnvironment

# Create an environment
#    this will use myImage.png as initial image to project
#    it will create a folder with the name of the image and take a snapshot of the flat wall
#    to get a 'label' image
env = ProjectorEnvironment("myImage.png")

# Set a new Image to project
# param new_class:
#    true: new image creates a new folder and label snapshot in that folder (treated as a new class)
#    false: adds the image to the current folder (treated as the same class)
env.NewImage(filepath, new_class=false)

# Create a new random wall to project image onto
env.NewRandomWall()

# Flatten Wall
env.FlattenWall()

# Set wall position to your requirements
wall = {
      "bottomLeft": 0.0,
      "bottomRight": 0.0,
      "bottomMiddle": 0.0,
      "middleLeft": 0.0,
      "middleRight": 0.0,
      "middleMiddle": 0.0,
      "topLeft": 0.0,
      "topRight": 0.0,
      "topMiddle": 0.0
    }
env.SetWall(wall)

# Save a snapshot of current projection on the wall
#    saved into the folder of the original image as the folder name
env.Snapshot("snapShotName.png")

# Close the environment to free up memory
env.close()
```
