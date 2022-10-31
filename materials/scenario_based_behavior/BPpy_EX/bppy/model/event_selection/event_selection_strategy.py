from abc import ABC, abstractmethod


class EventSelectionStrategy(ABC):

    @abstractmethod
    def select(self, statements):
        pass

    @abstractmethod
    def is_satisfied(self, event, statement):
        pass

    @abstractmethod
    def is_event_blocked(self, statements, bev):
        return False