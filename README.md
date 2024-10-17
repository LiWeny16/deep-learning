[TOC]
# UAV 3D Navigation Environment with Q-Learning

## Introduction

In this exciting project, we've created a 3D simulation for UAV (or drone) navigation. What sets this apart is the use of Q-learning, a fascinating branch of reinforcement learning. Our simulation platform, which is in Python, utilizes the `gymnasium` library, enabling our UAV to dodge obstacles and navigate to a certain destination.

## How the Code Works

### The UAV3DEnv Class

We've built the environment from the `UAV3DEnv` class. Here's what we do to navigate the drone.

- **State and Action Spaces**: The UAV knows its position in a three-dimensional grid, where it can execute one of five actions to move around.
- **Starts and Resets**: We start with a playing field scattered with random obstacles. Each episode resets the UAV to its starting point, ready to face a fresh layout.
- **Navigating Through Actions**: The step function is where the movement processed. Depending on the UAV's movement, it either gets closer to the goal, bumps into an obstacle, or maybe even hits the grid edge. We give penalties or positive rewards depending to the situations 

### The Learning Q-Learning Train Function

- **Q-Table**: Think of this as a cheat sheet that grows over time, where each correct or incorrect move updates our understanding of what works.
- **Learning Episodes**: Like replaying a tricky video game level, the UAV goes through multiple episodes, learning a bit more each time, tweaking its strategy using the Q-table insights. Taking random actions to explore the 3D map through `epsilon_decay` coefficient. 

## Tools We Used

To bring this environment to life, we leaned on several Python libraries:
- `gymnasium`: This library is the basic tool, offering a structured way to build and manage the learning landscape.
- `numpy`: Basic math tools
- `matplotlib`: Supporting UI as well as the animation graphic demo.

## Reflections/Lessons Learned:
- **Adjustments on the Fly**: Initially, we envisioned a more complex set of movements for the UAV, but quickly realized that starting simple was key to early learning. So we decrease 14 directions of the movement into only 6.
- **Finding Balance**: Tweaking the epsilon decay to balance between trying new moves and sticking to known paths was more art than science.
- **Always Improving**: Simple Q-learning is limited and we are going to use DQL in the next time.

## Conclusion

This project is a bit rough now and still has some problem especially in the accuracy of reaching the goal. However, we are going to improve our model in the next time with DQL. Hopes it will get better soon!
