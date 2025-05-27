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
  private Map<Integer, String> goalDescriptions; // Store goal descriptions for logging

  private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    // the URL of the W3C Thing Description of the lab Thing
    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n="+ stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m="+ actionCount);

    qTables = new HashMap<>();
    goalDescriptions = new HashMap<>();
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
    goalDescriptions.put(goalHash, Arrays.toString(goalDescription));

    int targetZ1 = Integer.parseInt(goalDescription[0].toString());
    int targetZ2 = Integer.parseInt(goalDescription[1].toString());
    
    LOGGER.info("Target goal: Z1=" + targetZ1 + ", Z2=" + targetZ2);

    // Enhanced training with convergence tracking
    int successfulEpisodes = 0;
    int consecutiveSuccesses = 0;
    double totalRewardLastHundred = 0.0;
    Queue<Double> recentEpisodeRewards = new LinkedList<>();

    for (int episodeNum = 0; episodeNum < totalEpisodes; episodeNum++) {
        initializeRandomState();
        int currentStateIdx = lab.readCurrentState();
        
        final int MAX_EPISODE_STEPS = 50; 
        boolean goalReached = false;
        double episodeReward = 0.0;
        
        for (int stepCount = 0; stepCount < MAX_EPISODE_STEPS; stepCount++) {
            List<Integer> validActions = lab.getApplicableActions(currentStateIdx);
            
            if (validActions.isEmpty()) {
                LOGGER.fine("No valid actions available at state " + currentStateIdx);
                break;
            }
            
            // Decay exploration rate over time for better convergence
            double currentEpsilon = explorationRate * Math.pow(0.995, episodeNum);
            int selectedAction = selectActionEpsilonGreedy(qMatrix, currentStateIdx, validActions, currentEpsilon);
            
            lab.performAction(selectedAction);
            int nextStateIdx = lab.readCurrentState();
            
            goalReached = isGoalState(goalDescription);
            
            double stepReward = computeRewardValue(goalDescription, null, goalReward);
            episodeReward += stepReward;
            
            double maxFutureQ = findMaxQValue(qMatrix, nextStateIdx, lab.getApplicableActions(nextStateIdx));
            double currentQValue = qMatrix[currentStateIdx][selectedAction];
            double updatedQValue = currentQValue + learningRate * (stepReward + discountFactor * maxFutureQ - currentQValue);
            qMatrix[currentStateIdx][selectedAction] = updatedQValue;

            currentStateIdx = nextStateIdx;

            if (goalReached) {
                successfulEpisodes++;
                consecutiveSuccesses++;
                LOGGER.fine("Goal reached in episode " + episodeNum + " at step " + stepCount);
                break;
            }
        }

        if (!goalReached) {
            consecutiveSuccesses = 0;
        }

        // Track recent performance
        recentEpisodeRewards.add(episodeReward);
        if (recentEpisodeRewards.size() > 100) {
            totalRewardLastHundred -= recentEpisodeRewards.poll();
        }
        totalRewardLastHundred += episodeReward;

        // Enhanced progress reporting
        if ((episodeNum + 1) % 50 == 0) {
            double successRate = (double) successfulEpisodes / (episodeNum + 1) * 100;
            double avgRecentReward = recentEpisodeRewards.size() > 0 ? totalRewardLastHundred / recentEpisodeRewards.size() : 0;
            
            LOGGER.info("Training progress: " + (episodeNum + 1) + "/" + totalEpisodes + 
                       " episodes | Success rate: " + String.format("%.1f%%", successRate) +
                       " | Avg recent reward: " + String.format("%.2f", avgRecentReward) +
                       " | Consecutive successes: " + consecutiveSuccesses);
            
            int currentState = lab.readCurrentState();
            lab.readCurrentState();
            if (lab.currentState != null && lab.currentState.size() >= 2) {
                LOGGER.info("Current state: Z1=" + lab.currentState.get(0) + ", Z2=" + lab.currentState.get(1));
            }
        }
        
        if (consecutiveSuccesses >= 10 && episodeNum > totalEpisodes / 4) {
            LOGGER.info("Early convergence detected after " + (episodeNum + 1) + " episodes");
            LOGGER.info("Achieved " + consecutiveSuccesses + " consecutive successes");
            break;
        }

        if (episodeNum > 0 && episodeNum % 100 == 0) {
            try {
                Thread.sleep(100); 
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    qTables.put(goalHash, qMatrix);
    
    double finalSuccessRate = (double) successfulEpisodes / totalEpisodes * 100;
    LOGGER.info("Goal: " + Arrays.toString(goalDescription));
    LOGGER.info("Final success rate: " + String.format("%.1f%%", finalSuccessRate));
    LOGGER.info("Total successful episodes: " + successfulEpisodes + "/" + totalEpisodes);
    
    displayQTableSample(qMatrix, goalDescription);
    
    logBestPolicy(qMatrix, goalDescription);
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
        LOGGER.severe("Available goals: " + goalDescriptions.keySet());
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
        LOGGER.warning("No applicable actions for current state " + currentStateIdx);
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
    
    LOGGER.info("FOR REAL LAB");
    LOGGER.info("Goal: " + Arrays.toString(goalDescription));
    LOGGER.info("Current state: " + Arrays.toString(currentStateDescription));
    LOGGER.info("State index: " + currentStateIdx);
    LOGGER.info("Selected action: " + bestAction.getActionTag());
    LOGGER.info("Action Q-value: " + String.format("%.3f", qMatrix[currentStateIdx][bestActionIdx]));
    LOGGER.info("Payload tags: " + Arrays.toString(bestAction.getPayloadTags()));
    LOGGER.info("Payload values: " + Arrays.toString(bestAction.getPayload()));

    LOGGER.info("Alternative actions and their Q-values:");
    for (int actionIdx : applicableActions) {
        if (actionIdx != bestActionIdx) {
            Action altAction = lab.getAction(actionIdx);
            if (altAction != null) {
                LOGGER.info("  " + altAction.getActionTag() + " -> Q=" + 
                           String.format("%.3f", qMatrix[currentStateIdx][actionIdx]));
            }
        }
    }
}

private boolean isGoalState(Object[] goalDescription) {
    try {
        int targetZ1Level = Integer.parseInt(goalDescription[0].toString());
        int targetZ2Level = Integer.parseInt(goalDescription[1].toString());

        lab.readCurrentState();
        List<Integer> stateVector = lab.currentState;

        if (stateVector == null || stateVector.size() < 2) {
            return false;
        }

        int actualZ1Level = stateVector.get(0);
        int actualZ2Level = stateVector.get(1);

        boolean goalAchieved = (actualZ1Level == targetZ1Level && actualZ2Level == targetZ2Level);
        
        if (goalAchieved) {
            LOGGER.fine("Goal state detected: Z1=" + actualZ1Level + ", Z2=" + actualZ2Level);
        }
        
        return goalAchieved;
    } catch (Exception e) {
        LOGGER.warning("Error checking goal state: " + e.getMessage());
        return false;
    }
}

private double computeRewardValue(Object[] goalDescription, List<Integer> goalStates, double goalReward) {
    try {
        int targetZ1Level = Integer.parseInt(goalDescription[0].toString());
        int targetZ2Level = Integer.parseInt(goalDescription[1].toString());

        lab.readCurrentState();
        List<Integer> stateVector = lab.currentState;

        if (stateVector == null || stateVector.size() < 7) {
            return -1.0; 
        }

        int actualZ1Level = stateVector.get(0);
        int actualZ2Level = stateVector.get(1);
        boolean zone1LightOn = stateVector.get(2) == 1;
        boolean zone2LightOn = stateVector.get(3) == 1;
        boolean zone1BlindsUp = stateVector.get(4) == 1;
        boolean zone2BlindsUp = stateVector.get(5) == 1;
        int sunshineLevel = stateVector.get(6);

        int prevZ1 = previousIlluminanceLevels.getOrDefault("Z1", actualZ1Level);
        int prevZ2 = previousIlluminanceLevels.getOrDefault("Z2", actualZ2Level);

        previousIlluminanceLevels.put("Z1", actualZ1Level);
        previousIlluminanceLevels.put("Z2", actualZ2Level);

        double totalReward = -0.01;

        boolean goalAchieved = (actualZ1Level == targetZ1Level && actualZ2Level == targetZ2Level);
        
        if (goalAchieved) {
            totalReward += goalReward;
            LOGGER.fine("GOAL ACHIEVED! Z1=" + actualZ1Level + ", Z2=" + actualZ2Level + 
                       " matches target [" + targetZ1Level + "," + targetZ2Level + "]");
            return totalReward; 
        }
        int z1Distance = Math.abs(actualZ1Level - targetZ1Level);
        int z2Distance = Math.abs(actualZ2Level - targetZ2Level);
        double totalDistance = z1Distance + z2Distance;
        
        if (totalDistance > 0) {
            double proximityReward = goalReward * 0.2 / (1 + totalDistance);
            totalReward += proximityReward;
        }

        if (zone1LightOn) totalReward -= 0.5;
        if (zone2LightOn) totalReward -= 0.5;
        if (zone1BlindsUp) totalReward -= 0.01;
        if (zone2BlindsUp) totalReward -= 0.01;
    
        int z1Change = Math.abs(actualZ1Level - prevZ1);
        int z2Change = Math.abs(actualZ2Level - prevZ2);
        totalReward -= 0.05 * (z1Change + z2Change);
      
        if (sunshineLevel >= 2) {
            if (zone1BlindsUp && zone1LightOn) {
                totalReward -= 0.3; 
            }
            if (zone2BlindsUp && zone2LightOn) {
                totalReward -= 0.3;
            }
        }

        return totalReward;
        
    } catch (Exception e) {
        LOGGER.warning("Error computing reward: " + e.getMessage());
        return -1.0;
    }
}

private void logBestPolicy(double[][] qMatrix, Object[] goalDescription) {
    
    List<StateActionPair> bestPairs = new ArrayList<>();
    for (int state = 0; state < qMatrix.length; state++) {
        for (int action = 0; action < qMatrix[state].length; action++) {
            if (qMatrix[state][action] > 0) {
                bestPairs.add(new StateActionPair(state, action, qMatrix[state][action]));
            }
        }
    }
    
    bestPairs.sort((a, b) -> Double.compare(b.qValue, a.qValue));
    
    LOGGER.info("Top 5 state-action pairs for goal " + Arrays.toString(goalDescription) + ":");
    
    for (int i = 0; i < Math.min(5, bestPairs.size()); i++) {
        StateActionPair pair = bestPairs.get(i);
        Action action = lab.getAction(pair.actionIndex);
        if (action != null) {
            LOGGER.info("  State " + pair.stateIndex + " -> " + action.getActionTag() + 
                       " (Q=" + String.format("%.3f", pair.qValue) + ")");
        }
    }
}

private static class StateActionPair {
    int stateIndex;
    int actionIndex;
    double qValue;
    
    StateActionPair(int state, int action, double q) {
        this.stateIndex = state;
        this.actionIndex = action;
        this.qValue = q;
    }
}

    /**
    * Print the Q matrix
    *
    * @param qTable the Q matrix
    */
void printQTable(double[][] qTable) {
    System.out.println("Q-MATRIX SUMMARY");
    int statesToShow = Math.min(10, qTable.length);
    
    for (int i = 0; i < statesToShow; i++) {
        System.out.print("State " + String.format("%3d", i) + ": [");
        for (int j = 0; j < qTable[i].length; j++) {
            if (qTable[i][j] != 0.0) {
                System.out.printf("%6.2f", qTable[i][j]);
            } else {
                System.out.print("  0.00");
            }
            if (j < qTable[i].length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }
    
    if (qTable.length > 10) {
        System.out.println("... (" + (qTable.length - 10) + " more states not shown)");
    }
    int nonZeroCount = 0;
    double maxQ = Double.NEGATIVE_INFINITY;
    double minQ = Double.POSITIVE_INFINITY;
    double sumQ = 0.0;
    
    for (int i = 0; i < qTable.length; i++) {
        for (int j = 0; j < qTable[i].length; j++) {
            if (qTable[i][j] != 0.0) {
                nonZeroCount++;
                maxQ = Math.max(maxQ, qTable[i][j]);
                minQ = Math.min(minQ, qTable[i][j]);
                sumQ += qTable[i][j];
            }
        }
    }
    
    if (nonZeroCount > 0) {
        System.out.println("Q-Table Statistics:");
        System.out.println("  Non-zero entries: " + nonZeroCount + "/" + (qTable.length * qTable[0].length));
        System.out.println("  Max Q-value: " + String.format("%.3f", maxQ));
        System.out.println("  Min Q-value: " + String.format("%.3f", minQ));
        System.out.println("  Average Q-value: " + String.format("%.3f", sumQ / nonZeroCount));
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
    try {
        Random rng = new Random();
        int numRandomActions = rng.nextInt(3) + 1;
        
        for (int i = 0; i < numRandomActions; i++) {
            int currentState = lab.readCurrentState();
            List<Integer> availableActions = lab.getApplicableActions(currentState);
            if (!availableActions.isEmpty()) {
                int randomActionIdx = rng.nextInt(availableActions.size());
                lab.performAction(availableActions.get(randomActionIdx));
                
                Thread.sleep(10);
            }
        }
    } catch (Exception e) {
        LOGGER.warning("Error initializing random state: " + e.getMessage());
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
    LOGGER.warning("Using fallback action due to error");
    actionTag.set("http://example.org/was#SetZ1Light");
    payloadTags.set(new Object[]{"Z1Light"});
    payload.set(new Object[]{true});
}

private void displayQTableSample(double[][] qMatrix, Object[] goalDescription) {
    LOGGER.info("Q-TABLE TRAINING RESULTS");
    LOGGER.info("Goal: " + Arrays.toString(goalDescription));
    
    int samplesToShow = Math.min(5, qMatrix.length);
    for (int i = 0; i < samplesToShow; i++) {
        StringBuilder sb = new StringBuilder();
        sb.append("State ").append(String.format("%3d", i)).append(": [");
        for (int j = 0; j < qMatrix[i].length; j++) {
            sb.append(String.format("%6.2f", qMatrix[i][j]));
            if (j < qMatrix[i].length - 1) sb.append(", ");
        }
        sb.append("]");
        LOGGER.info(sb.toString());
    }
    
    // Find and display best Q-values
    double maxQ = Double.NEGATIVE_INFINITY;
    int bestState = -1, bestAction = -1;
    int positiveQCount = 0;
    
    for (int i = 0; i < qMatrix.length; i++) {
        for (int j = 0; j < qMatrix[i].length; j++) {
            if (qMatrix[i][j] > maxQ) {
                maxQ = qMatrix[i][j];
                bestState = i;
                bestAction = j;
            }
            if (qMatrix[i][j] > 0) {
                positiveQCount++;
            }
        }
    }
    
    if (bestState >= 0) {
        Action bestActionObj = lab.getAction(bestAction);
        String actionName = bestActionObj != null ? bestActionObj.getActionTag() : "Unknown";
        
        LOGGER.info("Training Results Summary:");
        LOGGER.info("  Highest Q-value: " + String.format("%.3f", maxQ));
        LOGGER.info("  Best state-action: State " + bestState + " -> " + actionName);
        LOGGER.info("  Positive Q-values: " + positiveQCount + "/" + 
                   (qMatrix.length * qMatrix[0].length) + " entries");
        
        if (maxQ > 50.0) {
            LOGGER.info("  STATUS: Good convergence achieved!");
        } else if (maxQ > 10.0) {
            LOGGER.info("  STATUS: Moderate convergence - consider more episodes");
        } else {
            LOGGER.warning("  STATUS: Poor convergence - check reward function and parameters");
        }
    }
}

@OPERATION
public void getQTableStatus(Object[] goalDescription, OpFeedbackParam<String> status) {
    int goalKey = generateGoalKey(goalDescription);
    
    if (!qTables.containsKey(goalKey)) {
        status.set("No Q-table found for goal " + Arrays.toString(goalDescription));
        return;
    }
    
    double[][] qMatrix = qTables.get(goalKey);
    
    double maxQ = Double.NEGATIVE_INFINITY;
    int positiveCount = 0;
    
    for (int i = 0; i < qMatrix.length; i++) {
        for (int j = 0; j < qMatrix[i].length; j++) {
            maxQ = Math.max(maxQ, qMatrix[i][j]);
            if (qMatrix[i][j] > 0) positiveCount++;
        }
    }
    
    String statusMsg = String.format("Goal %s: Max Q=%.3f, Positive entries=%d, Ready for real lab: %s",
        Arrays.toString(goalDescription), maxQ, positiveCount, 
        maxQ > 10.0 ? "YES" : "RECOMMEND MORE TRAINING");
    
    status.set(statusMsg);
    LOGGER.info("Q-Table Status: " + statusMsg);
}
}