//illuminance controller agent - Enhanced with Real Lab Support

/*
* The URL of the W3C Web of Things Thing Description (WoT TD) of a lab environment
* Simulated lab WoT TD: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl"
* Real lab WoT TD: Get in touch with us by email to acquire access to it!
 */

/* Initial beliefs and rules */

// the agent has a belief about the location of the W3C Web of Thing (WoT) Thing Description (TD)
// that describes a lab environment to be learnt
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").

real_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab-real.ttlE").

// the agent believes that the task that takes place in the 1st workstation requires an indoor illuminance
// level of Rank 2, and the task that takes place in the 2nd workstation requires an indoor illumincance 
// level of Rank 3. Modify the belief so that the agent can learn to handle different goals.
task_requirements([2,3]).

learning_episodes(200).       // Increased episodes for better learning
learning_alpha(0.2).          
learning_gamma(0.8).         
learning_epsilon(0.3).       // Reduced epsilon for less exploration
goal_reward(100.0).          // Increased reward for better convergence

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
+!start : learning_lab_environment(SimUrl) 
  & real_lab_environment(RealUrl)
  & task_requirements([Z1Level, Z2Level])
  & learning_episodes(Episodes)
  & learning_alpha(Alpha)
  & learning_gamma(Gamma)
  & learning_epsilon(Epsilon)
  & goal_reward(Reward) <-

 .print("ILLUMINANCE CONTROLLER AGENT STARTING");
 .print("Target goal: Z1Level=", Z1Level, " and Z2Level=", Z2Level);
  
  
  // creates a QLearner artifact for learning the lab Thing described by the W3C WoT TD located at URL
  makeArtifact("qlearner", "tools.QLearner", [SimUrl], QLArtId);

  // creates a ThingArtifact artifact for reading and acting on the state of the lab Thing
  makeArtifact("sim_lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [SimUrl], SimLabArtId);
  
  Goal = [Z1Level, Z2Level];
  
  // Learn Q-table with proper parameters
  .print("Learning Q-table for goal ", Goal, "...");
  calculateQ(Goal, Episodes, Alpha, Gamma, Epsilon, Reward)[artifact_id("qlearner")];
  .print("Learning complete!");
  
  !test_policy_simulation(Goal, 0);
  
  !apply_to_real_lab(Goal, RealUrl).

+!test_policy_simulation(Goal, Step) : Step < 10 <-
  .print("\n--- Testing Step ", Step, " (Simulation) ---");
  
  readProperty("https://example.org/was#Status", Tags, Values)[artifact_id("sim_lab")];
  
  !find_value(Tags, Values, "Z1Level", Z1Val);
  !find_value(Tags, Values, "Z2Level", Z2Val);
  !find_value(Tags, Values, "Z1Light", Z1L);
  !find_value(Tags, Values, "Z2Light", Z2L);
  !find_value(Tags, Values, "Z1Blinds", Z1B);
  !find_value(Tags, Values, "Z2Blinds", Z2B);
  !find_value(Tags, Values, "Sunshine", Sun);

  CurrentState = [Z1Val, Z2Val, Z1L, Z2L, Z1B, Z2B, Sun];
  .print("Simulation state: ", CurrentState);
  .print("Z1 Level: ", Z1Val, " lux, Z2 Level: ", Z2Val, " lux");

  !check_goal(Goal, Z1Val, Z2Val, Done);

  if(Done) {
    .print("Moving to real lab application...");
  } else {
    getActionFromState(Goal, CurrentState, ActionTag, PayloadTags, PayloadValues)[artifact_id("qlearner")];
    .print("Simulation action: ", ActionTag);
    !execute_action_simulation(ActionTag, PayloadTags, PayloadValues);
    .wait(2000);
    !test_policy_simulation(Goal, Step+1);
  }.

+!test_policy_simulation(Goal, Step) <-
  .print("Simulation test completed after ", Step, " steps.").

+!apply_to_real_lab(Goal, RealUrl) <-
  .print("Connecting to real laboratory at: ", RealUrl);

  if(RealUrl == "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab-real.ttlE") {
    .print("ERROR: Real lab URL not configured!");

  } else {

    makeArtifact("real_lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [RealUrl], RealLabArtId);
    .print("Connected to real laboratory successfully!");
    

    !control_real_lab(Goal, 0);
  }.

+!control_real_lab(Goal, Step) : Step < 20 <-
  .print("\n REAL LAB CONTROL - Step ", Step, );

  readProperty("https://example.org/was#Status", Tags, Values)[artifact_id("real_lab")];
  .print("Real lab raw data - Tags: ", Tags);
  .print("Real lab raw data - Values: ", Values);

  !find_value(Tags, Values, "Z1Level", Z1Val);
  !find_value(Tags, Values, "Z2Level", Z2Val);
  !find_value(Tags, Values, "Z1Light", Z1L);
  !find_value(Tags, Values, "Z2Light", Z2L);
  !find_value(Tags, Values, "Z1Blinds", Z1B);
  !find_value(Tags, Values, "Z2Blinds", Z2B);
  !find_value(Tags, Values, "Sunshine", Sun);

  CurrentState = [Z1Val, Z2Val, Z1L, Z2L, Z1B, Z2B, Sun];
  .print("REAL LAB Current state: ", CurrentState);
  .print("REAL LAB Z1 Level: ", Z1Val, " lux, Z2 Level: ", Z2Val, " lux");

  !check_goal(Goal, Z1Val, Z2Val, GoalAchieved);

  if(GoalAchieved) {
    .print("Target Z1Level=", Goal[0], ", Z2Level=", Goal[1]);
    .print("Actual Z1Level=", Z1Val, " lux, Z2Level=", Z2Val, " lux");
  } else {
    .print("Getting next action from learned Q-table...");
    getActionFromState(Goal, CurrentState, ActionTag, PayloadTags, PayloadValues)[artifact_id("qlearner")];
    
    .print("REAL LAB Action: ", ActionTag);
    .print("REAL LAB Payload tags: ", PayloadTags);
    .print("REAL LAB Payload values: ", PayloadValues);
    
    !execute_action_real_lab(ActionTag, PayloadTags, PayloadValues);
    
    .print("Waiting for real lab to stabilize...");
    .wait(5000);
    !control_real_lab(Goal, Step+1);
  }.

+!control_real_lab(Goal, Step) <-
  .print("Real lab control completed after ", Step, " steps.").


+!execute_action_simulation(ActionTag, [PayloadTag], [PayloadValue]) <-
  .print("Executing simulation action: ", ActionTag);
  .term2string(PayloadTag, TagStr);
  
  if(.substring(TagStr, CleanTag, 1, .length(TagStr) - 1)) {
    FinalTag = CleanTag;
  } else {
    FinalTag = TagStr;
  };
  
  Payload = json([keyvalue(FinalTag, PayloadValue)]);
  invokeAction(ActionTag, _, Payload)[artifact_id("sim_lab")];
  .print("Simulation action executed successfully").

+!execute_action_real_lab(ActionTag, [PayloadTag], [PayloadValue]) <-
  .print("Executing action: ", ActionTag);
  .print("Setting ", PayloadTag, " to ", PayloadValue);
  
  .term2string(PayloadTag, TagStr);
  
  if(.substring(TagStr, CleanTag, 1, .length(TagStr) - 1)) {
    FinalTag = CleanTag;
  } else {
    FinalTag = TagStr;
  };
  
  Payload = json([keyvalue(FinalTag, PayloadValue)]);
  .print("REAL LAB Final payload: ", Payload);
  
  invokeAction(ActionTag, _, Payload)[artifact_id("real_lab")];
  .print("*** REAL LAB ACTION EXECUTED SUCCESSFULLY ***").

+!execute_action_real_lab(ActionTag, PayloadTags, PayloadValues) <-
  .print("ERROR: Unexpected real lab payload format");
  .print("ActionTag: ", ActionTag);
  .print("PayloadTags: ", PayloadTags);
  .print("PayloadValues: ", PayloadValues).

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
    .print("Goal check: Z1=", D1, " (target=", T1, "), Z2=", D2, " (target=", T2, "), Achieved: ", Done).

+!find_value([], [], Tag, Value) <-
    .print("WARNING: Tag ", Tag, " not found, using default value 0");
    Value = 0.

+!find_value([H_tag|T_tags], [H_val|T_vals], Tag, Value) <-
    .term2string(H_tag, TagStr);
    if(.sub_string(TagStr, Tag)) {
        Value = H_val;
        .print("Found ", Tag, " = ", Value);
    } else {
        !find_value(T_tags, T_vals, Tag, Value);
    }.
