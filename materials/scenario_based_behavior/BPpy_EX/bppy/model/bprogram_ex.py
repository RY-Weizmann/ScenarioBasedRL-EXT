from importlib import import_module
from inspect import getmembers, isfunction
from itertools import tee

from z3 import *
from bppy import *

class BProgram_ex(BProgram):

    def __init__(self, bthreads=None, source_name=None, event_selection_strategy=None, listener=None, infinite_run=False, bthreads_prefix='SBP_'):
        super().__init__(bthreads, source_name, event_selection_strategy, listener)
        self.infinite_run = infinite_run
        self.bthreads_prefix = bthreads_prefix
        self.bprogram_mini_step = 0

    def setup(self):
        if self.source_name:
            self.bthreads = [o[1]() for o in getmembers(import_module(self.source_name)) if
                             isfunction(o[1]) and o[1].__module__ == self.source_name and o[0].startswith(self.bthreads_prefix)] # Only difference from Super.setup is in this line here

            self.variables = dict([o for o in getmembers(import_module(self.source_name)) if
                                   isinstance(o[1], ExprRef) or isinstance(o[1], list)])

        self.tickets = [{'bt': bt} for bt in self.bthreads]
        self.advance_bthreads(None)

    def next_event(self):
        return(super().next_event())


    def is_event_blocked(self, bev):
        t_ev = BEvent(bev.name)
        return self.event_selection_strategy.is_event_blocked(self.tickets, t_ev)

    def one_pass_all_bthreads(self, single_pass, ev=None):
        # Main loop
        ## interrupted = False
        self.bprogram_mini_step +=1
        print("bprogram_mini_step {}".format(self.bprogram_mini_step))
        while not self.interrupted:
            if not ev is None:
                event = ev
                ev = None
            else:
                event = self.next_event()
            # Finish the program if no event is selected
            if event is None:
                if not self.infinite_run:
                    self.interrupted = True
                    break
                else:
                    event = None

            if self.listener:
                # self.interrupted = self.listener.event_selected(b_program=self, event=event)
                self.listener.event_selected(b_program=self, event=event)
            if not event is None: #7-6-2022
                self.advance_bthreads(event)

            if single_pass:
                break

        if self.interrupted:
            if self.listener:
                self.listener.ended(b_program=self)

        return self.interrupted

    def superstep_all_bthreads(self, ev=None):
        while True:
            self.bprogram_mini_step += 1
            print("bprogram_mini_step {}".format(self.bprogram_mini_step))
            if not ev is None:
                event = ev
                ev = None
            else:
                event = self.next_event()
            #7-6-2022 trying to avoid running with None self.advance_bthreads(event)#purpose of running here is to drive actuator b-threads!

            # Finish the superstep if no event is selected
            if event is None:
                    break
            self.advance_bthreads(event)  #7-6-2022 trying to avoid running with None WAS: # purpose of running here is to drive actuator b-threads!

            if self.listener:
                self.listener.event_selected(b_program=self, event=event)
            # self.advance_bthreads(event) # wrong in here, b-threads will stall !

        return False

    def initiate_run(self):
        self.interrupted = False
        if self.listener:
            self.listener.starting(b_program=self)

        self.setup()

        # if not single_pass:
        #     self.one_pass_all_bthreads(single_pass)

    def run(self, single_pass=False):
        self.interrupted = False
        if self.listener:
            self.listener.starting(b_program=self)

        self.setup()

        if not single_pass:
            self.one_pass_all_bthreads(single_pass)
