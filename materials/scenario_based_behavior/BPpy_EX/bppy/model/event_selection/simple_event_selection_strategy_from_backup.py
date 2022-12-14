from bppy.model.event_selection.event_selection_strategy import EventSelectionStrategy
import random


class SimpleEventSelectionStrategy(EventSelectionStrategy):

    def is_satisfied(self, event, statement):
        return statement.get('request') == event or statement.get('waitFor') == event

    def selectable_events(self, statements):
        possible_events = set()
        for statement in statements:
            if 'request' in statement:  # should be eligible for sets
                possible_events.add(statement['request'])
        for statement in statements:
            if 'block' in statement:
                possible_events.discard(statement['block'])
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
                    if cev == bev:
                        return True
                else:
                    if bev in statement.get('block'):
                        return True
        return False
