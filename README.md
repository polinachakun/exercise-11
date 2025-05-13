# Exercise 11: Reinforcement Learning Agents

This repository contains a partial implementation of a [JaCaMo](https://jacamo-lang.github.io/) application where a reinforcement learning agent operates in an unknown environment.

## Table of Contents
- [Project structure](#project-structure)
- [How to set up the simulated environment](#how-to-set-up-the-simulator)
- [Task 2](#task-2)
  - [Task 2.1](#task-21)
  - [Task 2.2](#task-22)
  - [Task 2.3](#task-23)
- [How to run the project](#how-to-run-the-project)
 - [Bonus: Learning and acting on the real environment](#bonus-learning-and-acting-on-the-real-environment)

## Project structure
```bash
├── simulator
│   └── simulator_flow.json # a simulator for the lab (smart factory) environment 
├── src
│   ├── agt
│   │   └── illuminance_controller_agent.asl # agent program of the illuminance controller agent that is responsible for managing the indoor illuminance level based on task requirements
│   └── env
│       └── tools
│           ├── Action.java 
│           ├── Lab.java # Lab instances manage the state space and action space of a lab environment (simulated or real) - extends LearningEnvironment
│           ├── LearningEnvironment.java # an abstract class whose concrete classes help in learning environments
│           └── QLearner.java # artifact that can be used for performing Q learning in lab environments
└── task.jcm # the configuration file of the JaCaMo application
```

## How to set up the simulator
See instructions in [/simulator](/simulator).

## Task 2
### Task 2.1
Extend the operation `calculateQ` in [`QLearner.java`](src/env/tools/QLearner.java) that calculates a Q matrix against a goal description.
- HINTS: 
  - The method `initializeQTable` of the class `QLearner` can be used to initialize a Q-Table with Q values of 0.0.
  - A `QLearner` artifact is always initialized against an instance of the class [`Lab`](src/env/tools/Lab.java). The class `Lab` (and its superclass [`LearningEnvironment`](src/env/tools/LearningEnvironment.java)) offers methods that may be useful to you (you can also ignore or modify the methods). For example:
    - the method `readCurrentState` can be used to read the current state of the environment during training;
    - the method `performAction` can be used to perform an action on the environment during training;
    - the method `getApplicableActions` can be used to retrieve a list of actions that are applicable on a state of the environment;
    - the method `getCompatibleStates` can be used to retrieve a list of states that are compatible with a given substate;
    - the (private) method `discretizeLightLevel` discretizes the light level to values [0, 3];
    - the (private) method `discretizeSunshine` discretizes the sunshine level to values [0, 3].
  - It is advised that, in the beginning of each episode, the `QLearner` randomizes the state of the environment (e.g. using the method `performAction` of the class `Lab`), so that training is performed against different initial states. 

### Task 2.2
Modify the implementation in [`illuminance_controller_agent.asl`](src/agt/illuminance_controller_agent.asl) so that the agent calculates Q tables for its desired environment states.
- HINTS: 
  - Enable the agent to use the operation `calculateQ` that you implemented in [`QLearner.java`](src/env/tools/QLearner.java) for Task 2.1.
  - The agent holds an initial belief of the form `task_requirements([Z1Level, Z2Level])` (e.g. `task_requirements([3,3]`) that represents the required illuminance conditions for the tasks that take place in the environment. You can pass the list `[Z1Level, Z2Level]` as a parameter to the operation `calculateQ`. You can modify the values stored in the initial belief to train for different goal descriptions.

### Task 2.3
- Modify the operation `getActionFromState` in [`QLearner.java`](src/env/tools/QLearner.java) that returns information about the next best action based on a given current state, and a given desired state. Use the QTable that has been computed for the given desired state to extract the next best action from the current state.
- Modify the implementation in [`illuminance_controller_agent.asl`](src/agt/illuminance_controller_agent.asl) so that the agent uses a `ThingArtifact` and a `QLearner` artifact to take actions towards its goal. 
  - HINTS: 
    - For example, the agent could:
      1. Perceive the state of the environment using a `ThingArtifact`.
      2. If the perceived state does not match the desired state of the agent, the agent should get the next best action for its goal and perceived state using a `QLearner` artifact.
      3. Act by using the information about the next best action, and using a `ThingArtifact`.
      4. Repeat until the perceived state matches the desired state of the agent. 

## How to run the project
You can run the project directly in Visual Studio Code or from the command line with Gradle 8.5.
- In VSCode:  Click on the Gradle Side Bar elephant icon, and navigate through `GRADLE PROJECTS` > `exercise-11` > `Tasks` > `jacamo` > `task`.
- On MacOS and Linux run the following command:
```shell
./gradlew task
```
- On Windows run the following command:
```shell
.\gradlew.bat task
```

## Bonus: Learning and acting on the real environment
Get in touch with us by email to request the W3C Web of Things Thing Description (WoT TD) of the real lab environment! Then, simply update the implementation in [`illuminance_controller_agent.asl`](src/agt/illuminance_controller_agent.asl) so that the agent uses the WoT TD of the real lab environment instead of the WoT TD of the simulated environment. 
