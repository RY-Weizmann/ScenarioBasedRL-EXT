from bppy.model.event_selection.event_selection_strategy import EventSelectionStrategy
from bppy.model.b_event import BEvent
from bppy.model.event_set import EmptyEventSet
import random
from collections.abc import Iterable

BLOCKED_EV_DEBUG_PRINT = False #True

class SimpleEventSelectionStrategy(EventSelectionStrategy):

    def is_satisfied(self, event, statement):
        if isinstance(statement.get('request'), BEvent):
            if isinstance(statement.get('waitFor'), BEvent):
                return statement.get('request') == event or statement.get('waitFor') == event
            else:
                return statement.get('request') == event or statement.get('waitFor', EmptyEventSet()).__contains__(event)
        else:
            if isinstance(statement.get('waitFor'), BEvent):
                return statement.get('request', EmptyEventSet()).__contains__(event) or statement.get('waitFor') == event
            else:
                return statement.get('request', EmptyEventSet()).__contains__(event) or statement.get('waitFor', EmptyEventSet()).__contains__(event)

    def selectable_events(self, statements):
        possible_events = set()
        for statement in statements:
            if 'request' in str(statement):  # should be eligible for sets
                if isinstance(statement['request'], Iterable):
                    possible_events.update(statement['request'])
                elif isinstance(statement['request'], BEvent):
                    possible_events.add(statement['request'])
                else:
                    pass #raise TypeError("request parameter should be BEvent or iterable")
        for statement in statements:
            if 'block' in str(statement):
                if isinstance(statement.get('block'), BEvent):
                    if BLOCKED_EV_DEBUG_PRINT:
                        print('Blocked: {}'.format(statement.get('block')))
                    possible_events.discard(statement.get('block'))
                else:
                    if BLOCKED_EV_DEBUG_PRINT:
                        if len(possible_events) > 0:
                            l_ev_dbg = list(possible_events)
                            print('1st ev: {}, is blocked: {}'.format(l_ev_dbg[0], self.is_event_blocked(statements, l_ev_dbg[0])))
                            print('possible: # {}, {}'.format(len(possible_events), possible_events))
                    possible_events = {x for x in possible_events if x not in statement.get('block')}
                    if BLOCKED_EV_DEBUG_PRINT:
                        print('possible: # {}, {}'.format(len(possible_events), possible_events))
        return possible_events

    def select(self, statements):
        selectable_events = self.selectable_events(statements)
        if selectable_events:
            return random.choice(tuple(selectable_events))
        else:
            return None

    def is_event_blocked(self, statements, bev):
        for statement in statements:
            if 'block' in statement:
                if isinstance(statement.get('block'), BEvent):#one event
                    cev = statement.get('block')
                    # print('Blocked: {}'.format(cev))
                    if cev.name == bev.name:
                        return True
                else:
                    if bev.name in str(statement.get('block')):
                        return True
        return False
