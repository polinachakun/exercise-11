package tools;

import java.util.*;

/**
 * An abstract class for representing a learning environment which can be used
 * for Q learning
 *
 */
public abstract class LearningEnvironment {

  /**
   * The state space of the learning environment.
   * <p>
   * Each state is represented as a
   * list of integer values, e.g., [0,1,0,1,0,0,2].
   * </p>
   */
  protected Set<List<Integer>> stateSpace = new HashSet<>();

  /**
   * The action space of the learning environment.
   * <p>
   * Each action can be retrieved
   * through an integer key, where the keys have a range [0, M), where M is the
   * number of actions in the environment.
   * </p>
   */
  protected HashMap<Integer, Action> actionSpace = new HashMap<>();

  /**
   * The semantic types of the affordances that relate to the actions in the
   * environment (e.g., "http://example.org/was#SetZ1Light")
   */
  protected List<String> affordanceTypes = new ArrayList<>();

  /**
   * Returns the size of the state space
   *
   * @return the size
   */
  public int getStateCount() {
    return stateSpace.size();
  }

  /**
   * Returns the size of the action space
   *
   * @return the size
   */
  public int getActionCount() {
    return actionSpace.size();
  }

  /**
   * Returns an {@link Action} instance from the action space based on the given
   * key
   *
   * @param action the action key
   */
  public Action getAction(int action) {
    return actionSpace.get(action);
  }

  /**
   * Returns the states that are compatible to a given substate description.
   * <p>
   * E.g., if the substate description is [3,3], then compatible states are
   * all the states described as [3,3,_,_,...,_].
   * </p>
   * <p>
   * E.g., if the substate description is [3,3,false,false,true,true,2], then
   * only compatible state is described as [3,3,0,0,1,1,2].
   * </p>
   * <p>
   * Each compatible state is represented as an integer value that corresponds to
   * the position of the state in the state space set (e.g. the state
   * [0,0,0,0,0,0,1]
   * is represented by the value 1).
   * </p>
   *
   * @param stateDescription the description of the substate
   * @return the list of integer values that represent the compatible states
   */
  public abstract List<Integer> getCompatibleStates(List<Object> stateDescription);

  /**
   * Returns the current state of the environment.
   * <p>
   * The state is represented as
   * an integer value that corresponds to the position of the state in the state
   * space.
   * </p>
   *
   * @return the current state
   */
  public abstract int readCurrentState();

  /**
   * Returns the actions that are applicable in a given state.
   * <p>
   * Each action is represented as an integer value that corresponds to the
   * key of the action in the action space.
   * The given state is represented as an integer value that corresponds to the
   * the position of the state in the state space.
   * </p>
   *
   * @param state the state
   * @return the applicable actions
   */
  public abstract List<Integer> getApplicableActions(int state);

  /**
   * Performs an action in the environment.
   * <p>
   * The action is represented as an
   * integer value that corresponds to the key of the action in the action space.
   * </p>
   *
   * @param action the action
   */
  public abstract void performAction(int action);

}
