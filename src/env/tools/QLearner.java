package tools;

import java.util.*;
import java.util.logging.*;
import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;

public class QLearner extends Artifact {

  private Lab lab; // the lab environment that will be learnt 
  private int stateCount; // the number of possible states in the lab environment
  private int actionCount; // the number of possible actions in the lab environment
  private HashMap<Integer, double[][]> qTables; // a map for storing the qTables computed for different goals
  private Map<String, Integer> previousIlluminanceLevels;

  private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    // the URL of the W3C Thing Description of the lab Thing
    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n="+ stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m="+ actionCount);

    qTables = new HashMap<>();
    previousIlluminanceLevels = new HashMap<>();
    previousIlluminanceLevels.put("Z1", 0);
    previousIlluminanceLevels.put("Z2", 0);
  }
/**
* Computes a Q matrix for the state space and action space of the lab, and against
* a goal description. For example, the goal description can be of the form [z1level, z2Level],
* where z1Level is the desired value of the light level in Zone 1 of the lab,
* and z2Level is the desired value of the light level in Zone 2 of the lab.
* For exercise 11, the possible goal descriptions are:
* [0,0], [0,1], [0,2], [0,3], 
* [1,0], [1,1], [1,2], [1,3], 
* [2,0], [2,1], [2,2], [2,3], 
* [3,0], [3,1], [3,2], [3,3].
*
*<p>
* HINT: Use the methods of {@link LearningEnvironment} (implemented in {@link Lab})
* to interact with the learning environment (here, the lab), e.g., to retrieve the
* applicable actions, perform an action at the lab during learning etc.
*</p>
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  episodesObj the number of episodes used for calculating the Q matrix
* @param  alphaObj the learning rate with range [0,1].
* @param  gammaObj the discount factor [0,1]
* @param epsilonObj the exploration probability [0,1]
* @param rewardObj the reward assigned when reaching the goal state
**/
  @OPERATION
  public void calculateQ(Object[] goalDescription, Object episodesObj, Object alphaObj, Object gammaObj, Object epsilonObj, Object rewardObj) {
    int totalEpisodes = Integer.parseInt(episodesObj.toString());
    double learningRate = Double.parseDouble(alphaObj.toString());
    double discountFactor = Double.parseDouble(gammaObj.toString());
    double explorationRate = Double.parseDouble(epsilonObj.toString());
    double goalReward = Double.parseDouble(rewardObj.toString());
  
    LOGGER.info("Starting Q-Learning training with " + totalEpisodes + " episodes");
    LOGGER.info("Learning parameters: α=" + learningRate + ", γ=" + discountFactor + ", ε=" + explorationRate);

    double[][] qMatrix = createQTable();
    int goalHash = generateGoalKey(goalDescription);

    List<Object> goalPattern = Arrays.asList(goalDescription[0], goalDescription[1], null, null, null, null, null);
    List<Integer> goalStates = lab.getCompatibleStates(goalPattern);
    
    if (goalStates.isEmpty()) {
        LOGGER.severe("No goal states found for goal description: " + Arrays.toString(goalDescription));
        return;
    }
    
    LOGGER.info("Found " + goalStates.size() + " goal states for " + Arrays.toString(goalDescription));

    for (int episodeNum = 0; episodeNum < totalEpisodes; episodeNum++) {
        initializeRandomState();
        int currentStateIdx = lab.readCurrentState();
        final int MAX_EPISODE_STEPS = 100;
        
        for (int stepCount = 0; stepCount < MAX_EPISODE_STEPS; stepCount++) {
            List<Integer> validActions = lab.getApplicableActions(currentStateIdx);
            
            if (validActions.size() == 0) {
                break;
            }
            
            int selectedAction = selectActionEpsilonGreedy(qMatrix, currentStateIdx, validActions, explorationRate);
            lab.performAction(selectedAction);
            int nextStateIdx = lab.readCurrentState();
            double stepReward = computeRewardValue(goalDescription, goalStates, goalReward);
            
            double maxFutureQ = findMaxQValue(qMatrix, nextStateIdx, lab.getApplicableActions(nextStateIdx));
            double currentQValue = qMatrix[currentStateIdx][selectedAction];
            double updatedQValue = currentQValue + learningRate * (stepReward + discountFactor * maxFutureQ - currentQValue);
            qMatrix[currentStateIdx][selectedAction] = updatedQValue;

            currentStateIdx = nextStateIdx;

            if (goalStates.contains(currentStateIdx)) {
                break;
            }
        }

        if ((episodeNum + 1) % 100 == 0) {
            LOGGER.info("Training progress: " + (episodeNum + 1) + "/" + totalEpisodes + " episodes completed");
        }
    }

    qTables.put(goalHash, qMatrix);
    LOGGER.info("Q-Learning training completed for goal " + Arrays.toString(goalDescription));
    
    if (LOGGER.isLoggable(Level.INFO)) {
        displayQTableSample(qMatrix, goalStates);
    }
  }
/**
* Returns information about the next best action based on a provided state and the QTable for
* a goal description. The returned information can be used by agents to invoke an action 
* using a ThingArtifact.
*
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  currentStateDescription the current state e.g. [2,2,true,false,true,true,2]
* @param  nextBestActionTag the (returned) semantic annotation of the next best action, e.g. "http://example.org/was#SetZ1Light"
* @param  nextBestActionPayloadTags the (returned) semantic annotations of the payload of the next best action, e.g. [Z1Light]
* @param nextBestActionPayload the (returned) payload of the next best action, e.g. true
**/
  @OPERATION
  public void getActionFromState(Object[] goalDescription, Object[] currentStateDescription,
      OpFeedbackParam<String> nextBestActionTag, OpFeedbackParam<Object[]> nextBestActionPayloadTags,
      OpFeedbackParam<Object[]> nextBestActionPayload) {
         
    int goalKey = generateGoalKey(goalDescription);

    if (!qTables.containsKey(goalKey)) {
        LOGGER.severe("Q-table not found for goal: " + Arrays.toString(goalDescription));
        setFallbackAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
        return;
    }

    double[][] qMatrix = qTables.get(goalKey);
    int currentStateIdx = parseStateDescription(currentStateDescription);
    
    if (currentStateIdx < 0) {
        LOGGER.warning("Invalid state description: " + Arrays.toString(currentStateDescription));
        setFallbackAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
        return;
    }

    List<Integer> applicableActions = lab.getApplicableActions(currentStateIdx);
    
    if (applicableActions.isEmpty()) {
        LOGGER.warning("No applicable actions for current state");
        setFallbackAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
        return;
    }

    int bestActionIdx = findBestAction(qMatrix, currentStateIdx, applicableActions);
    Action bestAction = lab.getAction(bestActionIdx);
    
    if (bestAction == null) {
        LOGGER.severe("Action object not found for index: " + bestActionIdx);
        setFallbackAction(nextBestActionTag, nextBestActionPayloadTags, nextBestActionPayload);
        return;
    }

    nextBestActionTag.set(bestAction.getActionTag());
    nextBestActionPayloadTags.set(bestAction.getPayloadTags());
    nextBestActionPayload.set(bestAction.getPayload());
    
    LOGGER.info("Selected action for goal " + Arrays.toString(goalDescription) + 
               ": " + bestAction.getActionTag() + 
               " (Q-value: " + String.format("%.3f", qMatrix[currentStateIdx][bestActionIdx]) + ")");
  }
    /**
    * Print the Q matrix
    *
    * @param qTable the Q matrix
    */
  void printQTable(double[][] qTable) {
     System.out.println("Q matrix");
    int statesToShow = Math.min(10, qTable.length);
    
    for (int i = 0; i < statesToShow; i++) {
      System.out.print("From state " + i + ":  ");
     for (int j = 0; j < qTable[i].length; j++) {
      System.out.printf("%6.2f ", (qTable[i][j]));
      }
      System.out.println();
    }
    
    if (qTable.length > 10) {
        System.out.println("... (" + (qTable.length - 10) + " more states not shown)");
    }
  }

  private double[][] createQTable() {
    double[][] qMatrix = new double[this.stateCount][this.actionCount];
    for (int stateIdx = 0; stateIdx < stateCount; stateIdx++){
      for(int actionIdx = 0; actionIdx < actionCount; actionIdx++){
        qMatrix[stateIdx][actionIdx] = 0.0;
      }
    }
    return qMatrix;
  }

  private void initializeRandomState() {
      Random rng = new Random();
      int numRandomActions = rng.nextInt(6) + 3;
      
      for (int i = 0; i < numRandomActions; i++) {
          int currentState = lab.readCurrentState();
          List<Integer> availableActions = lab.getApplicableActions(currentState);
          if (!availableActions.isEmpty()) {
              int randomActionIdx = rng.nextInt(availableActions.size());
              lab.performAction(availableActions.get(randomActionIdx));
          }
      }
  }

  private int selectActionEpsilonGreedy(double[][] qMatrix, int state, List<Integer> validActions, double epsilon) {
      Random rng = new Random();

      if (rng.nextDouble() < epsilon) {
          return validActions.get(rng.nextInt(validActions.size()));
      }

      return findBestAction(qMatrix, state, validActions);
  }

  private int findBestAction(double[][] qMatrix, int state, List<Integer> validActions) {
      int optimalAction = validActions.get(0);
      double maxQValue = qMatrix[state][optimalAction];

      for (int actionIdx : validActions) {
          if (qMatrix[state][actionIdx] > maxQValue) {
              maxQValue = qMatrix[state][actionIdx];
              optimalAction = actionIdx;
          }
      }

      return optimalAction;
  }

  private double findMaxQValue(double[][] qMatrix, int state, List<Integer> validActions) {
      if (validActions.isEmpty()) {
          return 0.0;
      }

      double maxValue = Double.NEGATIVE_INFINITY;
      for (int actionIdx : validActions) {
          maxValue = Math.max(maxValue, qMatrix[state][actionIdx]);
      }

      return maxValue;
  }

  private double computeRewardValue(Object[] goalDescription, List<Integer> goalStates, double goalReward) {
      int targetZ1Level = Integer.parseInt(goalDescription[0].toString());
      int targetZ2Level = Integer.parseInt(goalDescription[1].toString());

      lab.readCurrentState();
      List<Integer> stateVector = lab.currentState;

      int actualZ1Level = stateVector.get(0);
      int actualZ2Level = stateVector.get(1);
      boolean zone1LightOn = stateVector.get(2) == 1;
      boolean zone2LightOn = stateVector.get(3) == 1;
      boolean zone1BlindsUp = stateVector.get(4) == 1;
      boolean zone2BlindsUp = stateVector.get(5) == 1;
      int sunshineLevel = stateVector.get(6);

      int prevZ1 = previousIlluminanceLevels.get("Z1");
      int prevZ2 = previousIlluminanceLevels.get("Z2");

      previousIlluminanceLevels.put("Z1", actualZ1Level);
      previousIlluminanceLevels.put("Z2", actualZ2Level);

      double totalReward = -0.1;

      if (actualZ1Level == targetZ1Level && actualZ2Level == targetZ2Level) {
          totalReward += goalReward;
      }

      if (zone1LightOn) totalReward -= 2.0;
      if (zone2LightOn) totalReward -= 2.0;
      if (zone1BlindsUp) totalReward -= 0.02;
      if (zone2BlindsUp) totalReward -= 0.02;

      int z1Change = Math.abs(actualZ1Level - prevZ1);
      int z2Change = Math.abs(actualZ2Level - prevZ2);
      totalReward -= 0.1 * (z1Change + z2Change);

      if (sunshineLevel >= 2) {
          if (zone1BlindsUp && zone1LightOn) {
              totalReward -= 1.0;
          }
          if (zone2BlindsUp && zone2LightOn) {
              totalReward -= 1.0;
          }
      }

      return totalReward;
  }

  private int generateGoalKey(Object[] goalDescription) {
      return Arrays.hashCode(goalDescription);
  }

  private int parseStateDescription(Object[] stateDescription) {
      try {
          if (stateDescription.length == 7) {
              List<Object> stateList = Arrays.asList(stateDescription);
              List<Integer> compatibleStates = lab.getCompatibleStates(stateList);
              
              if (!compatibleStates.isEmpty()) {
                  return compatibleStates.get(0);
              }
          }
          
          return lab.readCurrentState();
          
      } catch (Exception e) {
          LOGGER.warning("Error parsing state description: " + e.getMessage());
          return lab.readCurrentState();
      }
  }

  private void setFallbackAction(OpFeedbackParam<String> actionTag, 
                               OpFeedbackParam<Object[]> payloadTags,
                               OpFeedbackParam<Object[]> payload) {
      actionTag.set("http://example.org/was#SetZ1Light");
      payloadTags.set(new Object[]{"Z1Light"});
      payload.set(new Object[]{true});
  }

  private void displayQTableSample(double[][] qMatrix, List<Integer> goalStates) {
      LOGGER.info("Sample Q-values from trained table:");
      
      int samplesToShow = Math.min(3, qMatrix.length);
      for (int i = 0; i < samplesToShow; i++) {
          StringBuilder sb = new StringBuilder();
          sb.append("State ").append(i).append(": [");
          for (int j = 0; j < qMatrix[i].length; j++) {
              sb.append(String.format("%.2f", qMatrix[i][j]));
              if (j < qMatrix[i].length - 1) sb.append(", ");
          }
          sb.append("]");
          LOGGER.info(sb.toString());
      }
      
      double maxQ = Double.NEGATIVE_INFINITY;
      int bestState = -1, bestAction = -1;
      
      for (int i = 0; i < qMatrix.length; i++) {
          for (int j = 0; j < qMatrix[i].length; j++) {
              if (qMatrix[i][j] > maxQ) {
                  maxQ = qMatrix[i][j];
                  bestState = i;
                  bestAction = j;
              }
          }
      }
      
      if (bestState >= 0) {
          LOGGER.info("Highest Q-value: " + String.format("%.3f", maxQ) + 
                     " (State: " + bestState + ", Action: " + bestAction + ")");
      }
  }
}