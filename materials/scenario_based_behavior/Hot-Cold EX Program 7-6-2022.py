import bppy
import random as rand
from bppy import *

iteration_counter = 0

def add_hot_ex():
	while True:
		print('Starting add hot ex')
		yield {waitFor: BEvent("WATER_LOW")}
		yield {request: BEvent("HOT")}
		yield {request: BEvent("HOT")}
		yield {request: BEvent("HOT")}
		print('Finished add hot ex')

def add_cold_ex():
	while True:
		print('Starting add cold ex')
		yield {waitFor: BEvent("WATER_LOW")}
		yield {request: BEvent("COLD")}
		yield {request: BEvent("COLD")}
		yield {request: BEvent("COLD")}
		print('Finished add cold ex')

def control_temp_ex():
	while True:
		print('Starting control_temp ex')
		yield {waitFor: BEvent("COLD"), block: BEvent("HOT")}
		yield {waitFor: BEvent("HOT"), block: BEvent("COLD")}
		print('Finished control_temp ex')


def add_hot():
	yield {request: BEvent("HOT")}
	yield {request: BEvent("HOT")}
	yield {request: BEvent("HOT")}


def add_cold():
	yield {request: BEvent("COLD")}
	yield {request: BEvent("COLD")}
	yield {request: BEvent("COLD")}


def control_temp():
	while True:
		yield {waitFor: BEvent("COLD"), block: BEvent("HOT")}
		yield {waitFor: BEvent("HOT"), block: BEvent("COLD")}

if __name__ == "__main__":
	# initial_list_add_hot_cold = [add_hot(), add_cold(), control_temp()]
	initial_list_add_hot_cold_ex = [add_hot_ex(), add_cold_ex(), control_temp_ex()]

	b_program = BProgram_ex(bthreads=initial_list, 
							event_selection_strategy=SimpleEventSelectionStrategy(),
							listener=PrintBProgramRunnerListener(),
							infinite_run=True)

	b_program.run(True)

	while True:
		SBP_requested_ev = BEvent("WATER_LOW")
		iteration_counter += 1

        #  The event is requested externally, thus must check it is not blocked:
        if not b_program.is_event_blocked(SBP_requested_ev):
            SBP_terminated = b_program.one_pass_all_bthreads(True, SBP_requested_ev)

        assert not SBP_terminated, 'SBP: premature program termination'
        if b_program.is_event_blocked(SBP_requested_ev): ###  Mind that either it was already blocked, or became blocked in the
                                                         ###  call to one_pass_all_bthreads above
            print("In mainloop -- Blocked ev {}, iteration: {}".format(SBP_requested_ev, iteration_counter))

        SBP_terminated = b_program.superstep_all_bthreads()
        print("fininshed super-step {} in mainloop \n\n".format(iteration_counter))
