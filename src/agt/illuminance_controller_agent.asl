//illuminance controller agent

/*
* The URL of the W3C Web of Things Thing Description (WoT TD) of a lab environment
* Simulated lab WoT TD: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl"
* Real lab WoT TD: Get in touch with us by email to acquire access to it!
*/

/* Initial beliefs and rules */

// the agent has a belief about the location of the W3C Web of Thing (WoT) Thing Description (TD)
// that describes a lab environment to be learnt
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").

// the agent believes that the task that takes place in the 1st workstation requires an indoor illuminance
// level of Rank 2, and the task that takes place in the 2nd workstation requires an indoor illumincance 
// level of Rank 3. Modify the belief so that the agent can learn to handle different goals.
task_requirements([2,3]).

learning_episodes(10).       
learning_alpha(0.2).          
learning_gamma(0.8).         
learning_epsilon(0.4).       
goal_reward(50.0).           

/* Initial goals */
!start. // the agent has the goal to start

/* 
 * Plan for reacting to the addition of the goal !start
 * Triggering event: addition of goal !start
 * Context: the agent believes that there is a WoT TD of a lab environment located at Url, and that 
 * the tasks taking place in the workstations require indoor illuminance levels of Rank Z1Level and Z2Level
 * respectively
 * Body: (currently) creates a QLearnerArtifact and a ThingArtifact for learning and acting on the lab environment.
*/
@start
+!start : learning_lab_environment(Url) 
  & task_requirements([Z1Level, Z2Level])
  & learning_episodes(Episodes)
  & learning_alpha(Alpha)
  & learning_gamma(Gamma)
  & learning_epsilon(Epsilon)
  & goal_reward(Reward) <-

 .print("I want to achieve Z1Level=", Z1Level, " and Z2Level=",Z2Level);

  // creates a QLearner artifact for learning the lab Thing described by the W3C WoT TD located at URL
  makeArtifact("qlearner", "tools.QLearner", [Url], QLArtId);

  // creates a ThingArtifact artifact for reading and acting on the state of the lab Thing
  makeArtifact("lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [Url], LabArtId);
  
  .print("Artifacts created successfully");
  
  !learn_with_timeout([Z1Level, Z2Level]);
  .

+!learn_with_timeout(GoalDescription) : 
    learning_episodes(Episodes)
  & learning_alpha(Alpha)
  & learning_gamma(Gamma)
  & learning_epsilon(Epsilon)
  & goal_reward(Reward) <-
  
  .print("Starting Q-learning with TIMEOUT protection...");
  
  .wait(1000); 
  
  calculateQ(GoalDescription, Episodes, Alpha, Gamma, Epsilon, Reward);
  
  .print("Q-learning completed (or timed out)");
  
  !test_result(GoalDescription);
  .

+!test_result(GoalDescription) <-
  .print("Testing learned policy...");
  
  getActionFromState(GoalDescription, [0, 0, false, false, false, false, 1], ActionTag, PayloadTags, Payload);
  .print("Recommended action: ", ActionTag, " with payload: ", Payload);
  
  .

+!learn_with_timeout(GoalDescription) <-
  .print("ERROR: Q-learning failed or timed out");
  .print("This suggests there may be an infinite loop in the calculateQ method");
  .