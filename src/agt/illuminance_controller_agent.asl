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
  
  Goal = [Z1Level, Z2Level];
  
  // Learn Q-table with proper parameters
  .print("Learning...");
  calculateQ(Goal, Episodes, Alpha, Gamma, Epsilon, Reward)[artifact_id("qlearner")];
  .print("Learning complete!");
  
  !run_policy(Goal, 0).

+!discretize_light_level(Value, Level) <-
    if(Value < 50) { 
        Level = 0; 
    } elif(Value < 100) { 
        Level = 1; 
    } elif(Value < 300) { 
        Level = 2; 
    } else { 
        Level = 3; 
    }.


+!check_goal([T1,T2], Z1Val, Z2Val, Done) <-
    !discretize_light_level(Z1Val, D1);
    !discretize_light_level(Z2Val, D2);
    Done = (D1 == T1 & D2 == T2);
    .print("Current levels: Z1=", D1, " (target=", T1, "), Z2=", D2, " (target=", T2, "), Goal achieved: ", Done).


+!find_value([], [], Tag, Value) <-
    .print("ERROR: Tag ", Tag, " not found");
    Value = 0.

+!find_value([H_tag|T_tags], [H_val|T_vals], Tag, Value) <-
    .term2string(H_tag, TagStr);
    if(.sub_string(TagStr, Tag)) {
        Value = H_val;
    } else {
        !find_value(T_tags, T_vals, Tag, Value);
    }.

+!run_policy(Goal, Step) : Step < 30 <-
  .print("\n=== Step ", Step, " ===");

  readProperty("https://example.org/was#Status", Tags, Values)[artifact_id("lab")];
  .print("Raw tags: ", Tags);
  .print("Raw values: ", Values);

  !find_value(Tags, Values, "Z1Level", Z1Val);
  !find_value(Tags, Values, "Z2Level", Z2Val);
  !find_value(Tags, Values, "Z1Light", Z1L);
  !find_value(Tags, Values, "Z2Light", Z2L);
  !find_value(Tags, Values, "Z1Blinds", Z1B);
  !find_value(Tags, Values, "Z2Blinds", Z2B);
  !find_value(Tags, Values, "Sunshine", Sun);

  CurrentState = [Z1Val, Z2Val, Z1L, Z2L, Z1B, Z2B, Sun];
  .print("Current state: ", CurrentState);
  .print("Z1 Level: ", Z1Val, " lux, Z2 Level: ", Z2Val, " lux");

  !check_goal(Goal, Z1Val, Z2Val, Done);

  if(Done) {
    .print("*** GOAL ACHIEVED! ***");
  } else {
    .print("Getting next action from Q-learner...");
    getActionFromState(Goal, CurrentState, ActionTag, PayloadTags, PayloadValues)[artifact_id("qlearner")];
    
    .print("Recommended action: ", ActionTag);
    .print("Payload tags: ", PayloadTags);
    .print("Payload values: ", PayloadValues);
    
    !execute_action(ActionTag, PayloadTags, PayloadValues);
    
    .wait(3000);
    !run_policy(Goal, Step+1);
  }.

+!run_policy(Goal, Step) <-
  .print("Maximum steps (", Step, ") reached. Stopping execution.").

+!execute_action(ActionTag, [PayloadTag], [PayloadValue]) <-
  .print("Executing action: ", ActionTag);
  .print("Setting ", PayloadTag, " to ", PayloadValue);
  
  .term2string(PayloadTag, TagStr);
  
  if(.substring(TagStr, CleanTag, 1, .length(TagStr) - 1)) {
    FinalTag = CleanTag;
  } else {
    FinalTag = TagStr;
  };
  
  Payload = json([keyvalue(FinalTag, PayloadValue)]);
  .print("Final payload: ", Payload);
  
  invokeAction(ActionTag, _, Payload)[artifact_id("lab")];
  .print("Action executed successfully").

+!execute_action(ActionTag, PayloadTags, PayloadValues) <-
  .print("ERROR: Unexpected payload format");
  .print("ActionTag: ", ActionTag);
  .print("PayloadTags: ", PayloadTags);
  .print("PayloadValues: ", PayloadValues).

-!execute_action(ActionTag, PayloadTags, PayloadValues) <-
  .print("FAILED to execute action: ", ActionTag);
  .print("Tags: ", PayloadTags, ", Values: ", PayloadValues).

-!run_policy(Goal, Step) <-
  .print("ERROR in run_policy at step ", Step);
  .print("Retrying after 5 seconds...");
  .wait(5000);
  !run_policy(Goal, Step).

-!find_value(Tags, Values, Tag, Value) <-
  .print("ERROR finding value for tag: ", Tag);
  Value = 0.