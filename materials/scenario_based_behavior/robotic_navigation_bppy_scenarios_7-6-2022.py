import bppy
import random as rand
from bppy import *

iteration_counter = 0
stale_counter = 0  # For testing purpose
stale_action = None  # For testing purpose
def get_random_action():
    global stale_counter
    global stale_action
    act = random.uniform(0, 3)
    random_state = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 5)]

    if act < 1:
        requested_ev = BEvent("SBP_MoveForward")
    elif act < 2:
        requested_ev = BEvent("SBP_TurnRight")
    else:
        requested_ev = BEvent("SBP_TurnLeft")

    requested_ev.data['state'] = random_state
    if abs(act - 1.534) < 0.1:
        stale_counter = 8

    if stale_counter > 0:
        requested_ev = stale_action
        stale_counter -= 1
    else:
        stale_action = requested_ev
    # print(requested_ev)
    return requested_ev


FWD_DIR = 0.5
FWD_DIR_TOLERANCE = 0.1
MINIMAL_CLEARANCE = 0.30
MINIMAL_FWD_CLEARANCE = 0.35
# scenario 5: if target is straight ahead, move FWD !
def SBP_avoid_turning_when_clear():
    blockedEvList = []
    waitforEvList = [BEvent("SBP_MoveForward"), BEvent("SBP_TurnLeft"), BEvent("SBP_TurnRight")]
    while True:
        lastEv = yield {waitFor: waitforEvList, block: blockedEvList}
        # print("{} in SBP_avoid_turning_when_clear iteration: {}".format(lastEv.name, iteration_counter))
        if False and lastEv is None:
            print("None EV in SBP_avoid_turning_when_clear {iteration_counter}")
        else:
            state = lastEv.data['state']
            t_dir = state[-2]
            if state[3] > MINIMAL_FWD_CLEARANCE and state[2] > MINIMAL_CLEARANCE and state[4] > MINIMAL_CLEARANCE and abs(FWD_DIR - t_dir) < FWD_DIR_TOLERANCE: #all clear, goal ahead
                print("SBP_avoid_turning_when_clear dir: {} Blocking {} {} iteration: {}".format(t_dir, BEvent("SBP_TurnLeft"), BEvent("SBP_TurnRight"), iteration_counter))
                blockedEvList.extend([BEvent("SBP_TurnLeft"), BEvent("SBP_TurnRight")])
            else:
                blockedEvList = []
# The Python implementation of a scenario that blocks rotating if the goal is head-on and the path is clear.


def SBP_avoid_k_consecuative_turns():
    k = 7
    counter = 0
    prevEv = None
    blockedEvList = []
    waitforEvList = [BEvent("SBP_MoveForward"), BEvent("SBP_TurnLeft"), BEvent("SBP_TurnRight")]
    while True:
        lastEv = yield {waitFor: waitforEvList, block: blockedEvList}
        # print("{} in SBP_avoid_k_consecuative_turns iteration: {}".format(lastEv.name, iteration_counter))
        if False and lastEv is None:
            print("None in SBP_avoid_k_consecuative_turns iteration: {}".format(iteration_counter))
        else:
            if prevEv is None or lastEv.name == "SBP_MoveForward" or prevEv.name != lastEv.name:
                prevEv = lastEv
                counter = 0
                blockedEvList = []
            else:
                if counter == k - 1:
                    # Blocking!
                    print("SBP_avoid_k_consecuative_turns Blocking {}, iteration: {}".format(lastEv, iteration_counter))
                    blockedEvList.append(lastEv)
                else:
                    counter += 1
# The Python implementation of a scenario that blocks rotating in the same direction more then k − 1 consecutive times.


def SBP_avoidBackAndForthRotation():
    blockedEvList = []
    waitforEvList = [BEvent("SBP_MoveForward", None),
                     BEvent("SBP_TurnLeft", None),
                     BEvent("SBP_TurnRight", None)]
    while True:
        lastEv = yield {waitFor: waitforEvList, block: blockedEvList}
        # print("{} in SBP_avoidBackAndForthRotation iteration: {}".format(lastEv.name, iteration_counter))
        if False and lastEv is None:
            print("None in SBP_avoidBackAndForthRotation iteration: {}".format(iteration_counter))
        else:
            if lastEv.name != "SBP_TurnLeft" and lastEv.name != "SBP_TurnRight":
                blockedEvList = []
            else:
                blocked_ev = BEvent("SBP_TurnRight", None) if lastEv.name == "SBP_TurnLeft" else BEvent("SBP_TurnLeft", None)
                blocked_ev.data = None
                # Blocking!
                print("SBP_avoidBackAndForthRotation blocking {}, iteration: {}".format(blocked_ev, iteration_counter))
                blockedEvList.append(blocked_ev)


if __name__ == "__main__":
    initial_list = [SBP_avoid_turning_when_clear(), SBP_avoid_k_consecuative_turns(), SBP_avoidBackAndForthRotation()]
    b_program = BProgram_ex(bthreads=initial_list, #source_name='__main__',
                            event_selection_strategy=SimpleEventSelectionStrategy(),
                            listener=PrintBProgramRunnerListener(),
                            infinite_run=True)

    b_program.run(True)

    while True:
        SBP_requested_ev = get_random_action()
        iteration_counter += 1

        #  The event is requested externally, thus must check it is not blocked:
        if not b_program.is_event_blocked(SBP_requested_ev):
            SBP_terminated = b_program.one_pass_all_bthreads(True, SBP_requested_ev)

        assert not SBP_terminated, 'SBP: premature program termination'
        if b_program.is_event_blocked(SBP_requested_ev): ###  Mind that either it was already blocked, or became blocked in the
                                                         ###  call to one_pass_all_bthreads above
            print("In mainloop -- Blocked ev {}, iteration: {}".format(SBP_requested_ev, iteration_counter))
        else:
            SBP_terminated = b_program.superstep_all_bthreads()
        print("fininshed super-step {} in mainloop \n\n".format(iteration_counter))
