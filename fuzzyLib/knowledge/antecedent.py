from typing import Dict, Callable
from fuzzyLib.algebras.algebra import Algebra
from fuzzyLib.fuzzySetsFol.fuzzy_set import MembershipDegree
from fuzzyLib.knowledge.clause import Clause

from abc import ABC, abstractmethod


class Antecedent(ABC):
    """
    Base class for representing an antecedent:
    https://en.wikipedia.org/wiki/Fuzzy_set
    It is the abstract class that represent the method of how the activation of the interface would be callculated.
    It is used for creating term that represent the actual "everythin what is before implication symbol"
    """
    def __init__(self, algebra: Algebra):
        if not isinstance(algebra, Algebra):
            raise TypeError('Algebra must be of Algebra type')
        self.__algebra = algebra

    @property
    @abstractmethod
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]:
        pass

    @property
    def algebra(self) -> Algebra:
        """
        Getter of the algebra.

        :return: algebra
        """
        return self.__algebra
