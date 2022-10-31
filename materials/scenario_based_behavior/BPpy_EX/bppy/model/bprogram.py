from importlib import import_module
from inspect import getmembers, isfunction
from itertools import tee

from z3 import *


class BProgram:

    def __init__(self, bthreads=None, source_name=None, event_selection_strategy=None, listener=None):
        self.source_name = source_name
        self.bthreads = bthreads
        self.event_selection_strategy = event_selection_strategy
        self.listener = listener
        self.variables = None
        self.tickets = None

    def setup(self):
        if self.source_name:
            self.bthreads = [o[1]() for o in getmembers(import_module(self.source_name)) if
                             isfunction(o[1]) and o[1].__module__ == self.source_name]

            self.variables = dict([o for o in getmembers(import_module(self.source_name)) if
                                   isinstance(o[1], ExprRef) or isinstance(o[1], list)])

        self.tickets = [{'bt': bt} for bt in self.bthreads]
        self.advance_bthreads(None)

## Raz: I Suspect a defect inadvance_bthreads below, when there is a BT that only "request" some BEvent, and some BEvent is blocked, seems that BT does not get to run!
    def advance_bthreads(self, m):
        for l in self.tickets:
            if m is None or self.event_selection_strategy.is_satisfied(m, l):
            #7-6-2022 if self.event_selection_strategy.is_satisfied(m, l):
                try:
                    bt = l['bt']
                    l.clear()
                    ll = bt.send(m)
                    if ll is None:
                        continue
                    l.update(ll)
                    l.update({'bt': bt})
                except (KeyError, StopIteration):
                    pass

    def next_event(self):
        return self.event_selection_strategy.select(self.tickets)

    def run(self):
        if self.listener:
            self.listener.starting(b_program=self)

        self.setup()

        # Main loop
        interrupted = False
        while not interrupted:
            event = self.next_event()
            # Finish the program if no event is selected
            if event is None:
                break
            if self.listener:
                interrupted = self.listener.event_selected(b_program=self, event=event)
            self.advance_bthreads(event)

        if self.listener:
            self.listener.ended(b_program=self)


#reverting changes from bprogram, all included in bprogram_ex
    def _HIDE_is_event_blocked(self, bev):
        return self.event_selection_strategy.is_event_blocked(self.tickets, bev)

    def _HIDE_one_pass_all_bthreads(self, single_step):
        # Main loop
        ## interrupted = False
        while not self.interrupted:
            event = self.next_event()
            # Finish the program if no event is selected
            if event is None:
                self.interrupted = True
                break
            if self.listener:
                self.interrupted = self.listener.event_selected(b_program=self, event=event)
            self.advance_bthreads(event)

            if single_step:
                break

        if self.interrupted:
            if self.listener:
                self.listener.ended(b_program=self)

        return self.interrupted


    def _HIDE_run(self, single_step=False):
        self.interrupted = False
        if self.listener:
            self.listener.starting(b_program=self)

        self.setup()

        # Main loop
        # print("calling step")
        if not single_step:
            self.one_pass_all_bthreads(single_step)

        # interrupted = False
        # while not interrupted:
        #     event = self.next_event()
            ## Finish the program if no event is selected
            # if event is None:
            #     break
            # if self.listener:
            #     interrupted = self.listener.event_selected(b_program=self, event=event)
            # self.advance_bthreads(event)

        # if self.listener:
        #     self.listener.ended(b_program=self)
