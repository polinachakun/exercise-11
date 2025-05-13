package tools;

import java.util.Arrays;
import ch.unisg.ics.interactions.wot.td.clients.TDHttpRequest;

public class Action {

  private final String actionTag;
  private final Object[] payloadTags;
  private final Object[] payload;
  private final TDHttpRequest request;

  private int applicableOnStateAxis;
  private int applicableOnStateValue;

  public Action(String actionTag, Object[] payloadTags,
    Object[] payload, TDHttpRequest request) {
      this.actionTag = actionTag;
      this.payloadTags = payloadTags;
      this.payload = payload;
      this.request = request;
    }

  @Override
  public String toString() {
    return "Action Tag: " + this.actionTag +
    ", Payload Tags: " + Arrays.toString(this.payloadTags) +
    ", Payload: " + Arrays.toString(this.payload);
  }

  public String getActionTag() {
    return this.actionTag;
  }

  public Object[] getPayloadTags() {
    return this.payloadTags;
  }

  public Object[] getPayload() {
    return this.payload;
  }

  public TDHttpRequest getRequest() {
    return this.request;
  }

  public int getApplicableOnStateAxis() {
    return this.applicableOnStateAxis;
  }

  public int getApplicableOnStateValue() {
    return this.applicableOnStateValue;
  }

  public void setApplicableOn(int stateAxis, int stateValue) {
    this.applicableOnStateAxis = stateAxis;
    this.applicableOnStateValue = stateValue;
  }
}
