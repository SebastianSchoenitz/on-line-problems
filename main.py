from abc import ABC, abstractmethod
from typing import Any
from enum import Enum
import numpy as np


class Requests(ABC):
    """
    Abstract iterator class for requests
    """

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError("Abstract method must be implemented")

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Any:
        raise NotImplementedError("Abstract method must be implemented")


class Algorithm(ABC):
    """
    Abstract class for answers
    """

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError("Abstract method must be implemented")

    @abstractmethod
    def compute_answer(self, state) -> None:
        raise NotImplementedError("Abstract method must be implemented")


class Tracking(ABC):
    """
    Abstract class for tracking
    """

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError("Abstract method must be implemented")


class State(ABC):
    """
    Abstract class for state
    """

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError("Abstract method must be implemented")

    @abstractmethod
    def update_request(request):
        raise NotImplementedError("Abstract method must be implemented")

    @abstractmethod
    def update_answer(request):
        raise NotImplementedError("Abstract method must be implemented")


class Tracking(ABC):
    """
    Abstract class for tracking
    """

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError("Abstract method must be implemented")

    @abstractmethod
    def update_state(self) -> None:
        raise NotImplementedError("Abstract method must be implemented")

    @abstractmethod
    def update_answer(self) -> None:
        raise NotImplementedError("Abstract method must be implemented")


class SimpleTracking(Tracking):
    """ """

    def __init__(self) -> None:
        self.miss_counter = 0
        self.answers = []

    def update_state(self, state) -> None:
        if state.hit_miss == 1:
            self.miss_counter += 1

    def update_answer(self, answer) -> None:
        self.answer.append(answer)


class Environment(object):
    """
    Environment class
    """

    def __init__(self, requests, algorithms_and_states, tracking) -> Any:
        self._requests = requests
        self._algorithms_and_states = algorithms_and_states
        self._trackings = tracking  # [tracking for _ in self._algorithms_and_states]

    def run(self) -> None:
        for request in self._requests:
            for alg, state in self._algorithms_and_states:
                state.update_request(request)
                self._trackings.update_state(state)
                answer = alg.compute_answer(state)
                state.update_answer(answer)
                # self._tracking.update_answer(answer)


class SimpleRequests(Requests):
    """
    Cyclic requests
    """

    def __init__(self, n: int):
        self.start = 0
        self.stop = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.start >= self.stop:
            raise StopIteration
        else:
            i = self.start
            self.start += 1
            return i % 6


class RandomRequests(Requests):
    """
    Random requests
    """

    def __init__(self, n: int, seed: int):
        self.start = 0
        self.stop = n
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start >= self.stop:
            raise StopIteration
        else:
            self.start += 1
            return self.random_state.randint(low=0, high=6)


class SimpleCache(State):
    """
    Simple unoptimizied cache
    """

    def __init__(self, cache_size):
        self.cache = [None] * cache_size
        self.how_long = np.array([-1] * cache_size)
        self.newest_request = None
        self.hit_miss = 0

    def update_request(self, request):
        self.newest_request = request
        if self.newest_request in self.cache:
            self.hit_miss = 0
        else:
            self.hit_miss = 1

    def update_answer(self, answer):
        for idx, el in enumerate(self.how_long):
            if el > -1:
                self.how_long[idx] += 1
        if self.hit_miss == 1:
            self.cache[answer] = self.newest_request
            self.how_long[answer] = 0


class LifoAlgo(Algorithm):
    """
    Last in first out. Replace the last inserted page when a cache miss appears.
    """

    def __init__(self):
        pass

    def compute_answer(self, state):
        return np.argmin(state.how_long)


class FifoAlgo(Algorithm):
    """
    First in first. Replace the page which was the longest time in the cache when a cache miss appears.
    """

    def __init__(self):
        pass

    def compute_answer(self, state):
        if -1 in state.how_long:
            return np.argmin(state.how_long)
        else:
            return np.argmax(state.how_long)


def main():
    r = SimpleRequests(202)
    r = RandomRequests(202, 0)
    s = SimpleCache(4)
    lifo = LifoAlgo()
    fifo = FifoAlgo()
    t = SimpleTracking()
    e = Environment(r, [(fifo, s)], t)
    e.run()
    print(e._algorithms_and_states[0][1].cache)
    print(e._algorithms_and_states[0][1].how_long)
    print("Misses: ", e._trackings.miss_counter)


if __name__ == "__main__":
    main()
