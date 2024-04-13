from __future__ import annotations

from typing import Dict, Sequence, Callable, NoReturn
from fuzzyLib.algebras.algebra import Algebra
from fuzzyLib.fuzzySetsFol.fuzzy_set import MembershipDegree
from fuzzyLib.knowledge.clause import Clause
from fuzzyLib.knowledge.antecedent import Antecedent

from functools import partial


class Term(Antecedent):
    """
    Class representing an antecedent with recursive firing value computation:
    https://en.wikipedia.org/wiki/Fuzzy_set

    Attributes
    --------------------------------------------
    __algebra : Algebra
        algebra provides t-norm and s-norm

    Methods
    --------------------------------------------
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]
        returns a firing value of the antecedent

    Examples
    --------------------------------------------
    """

    def __init__(self, algebra: Algebra, clause: Clause = None, name: str = None):
        """
        Creates Term object with given algebra and clause.
        :param algebra: algebra provides t-norm and s-norm
        :param clause: provides a linguistic variable (temerature) with corresponding fuzzy set and gradiation adjective (low)
        :param name: name of the term
        """
        super().__init__(algebra)

        if not clause:
            self.__fire = None
            if name is None:
                self.name = ''
            else:
                self.name = name
        else:
            if name is None:
                self.name = clause.linguistic_variable.name + '_' + clause.gradation_adjective
            else:
                self.name = name
            self.__fire = partial(self.dict_clause, clause=clause)

    def dict_clause(self, dict_ : dict[Clause, MembershipDegree], clause: Clause):
        return dict_[clause]

    @property
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]:
        """
        Returns the firing function.
        :return: firing function
        
        It return the partial function dict_clause with set clause argument. That is why it return the function
        that after passing Dict[Clause, MembershipDegree] return MembershipDegree
        """
        return self.__fire

    @fire.setter
    def fire(self, fire: Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]):
        """
        Sets new firing function to the antecedent.

        :param fire: firing function
        """
        self.__fire = fire

    def __and__(self, other: Term) -> Term:
        """
        Creates new antecedent object and sets new firing function which uses t-norm.
        We create new term using t_norm operator between self: Term and the other: Term
        example:
        "temperature is high and humidity is low "

        :param other: other term
        :return: term
        """
        new_term = self.__class__(self.algebra, name=self.name + ' & ' + other.name)
        # Sprawdzić czy działa

        # now new fire is the partial of apply_dict_t_norm that also takes as arguments Dict[Clause, MembershipDegree]
        new_term.fire = partial(self.apply_dict_t_norm, other=other)
        return new_term

    def __or__(self, other: Term) -> Term:
        """
        Creates new antecedent object and sets new firing function which uses s-norm.
        We create new term using s_norm operator between self: Term and the other: Term
        example:
        "temperature is high or humidity is low "



        :param other: other term
        :return: term
        """
        new_term = self.__class__(self.algebra, name=self.name + ' | ' + other.name)
        # Sprawdzić czy działa
        new_term.fire = partial(self.apply_dict_s_norm, other=other)
        return new_term

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def apply_dict_t_norm(self, dict_, other):
        return self.algebra.t_norm(self.fire(dict_), other.fire(dict_))

    def apply_dict_s_norm(self, dict_, other):
        return self.algebra.s_norm(self.fire(dict_), other.fire(dict_))
