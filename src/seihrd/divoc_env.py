from seihrd.base_models import (
    State,
    Populations,
    SubCompartmentPopulations,
    Probs,
    SimHyperParams,
    A,
)
from random import random


class Transitions:
    def __init__(self, state: State):
        self.s = state.copy(deep=True)

    def step(self):
        self.vax()

        self.transition_compartment(
            population_from=self.s.populations.susceptible,
            population_to=self.s.populations.exposed1,
            prob=self.s.probs.s_e1,
        )
        self.transition_compartment(
            population_from=self.s.populations.exposed1,
            population_to=self.s.populations.susceptible,
            prob=self.s.probs.e1_s,
        )
        self.transition_compartment(
            population_from=self.s.populations.exposed1,
            population_to=self.s.populations.infected,
            prob=self.s.probs.e1_i,
        )
        self.transition_compartment(
            population_from=self.s.populations.infected,
            population_to=self.s.populations.hospitalized,
            prob=self.s.probs.i_h,
        )
        self.transition_compartment(
            population_from=self.s.populations.infected,
            population_to=self.s.populations.recovered,
            prob=self.s.probs.i_r,
        )
        self.transition_compartment(
            population_from=self.s.populations.infected,
            population_to=self.s.populations.deceased,
            prob=self.s.probs.i_d,
        )
        self.transition_compartment(
            population_from=self.s.populations.hospitalized,
            population_to=self.s.populations.recovered,
            prob=self.s.probs.h_r,
        )
        self.transition_compartment(
            population_from=self.s.populations.hospitalized,
            population_to=self.s.populations.deceased,
            prob=self.s.probs.h_d,
        )
        self.transition_compartment(
            population_from=self.s.populations.recovered,
            population_to=self.s.populations.exposed2,
            prob=self.s.probs.r_e2,
        )
        self.transition_compartment(
            population_from=self.s.populations.exposed2,
            population_to=self.s.populations.infected,
            prob=self.s.probs.e2_i,
        )
        self.transition_compartment(
            population_from=self.s.populations.exposed2,
            population_to=self.s.populations.recovered,
            prob=self.s.probs.e2_r,
        )
        return self.s

    def vax(self):
        for c in Populations.__fields__:
            population = getattr(self.s.populations, c)

            # uv to fv
            for i in range(population.uv):
                if random() < self.s.probs.vfv:
                    population.uv -= 1
                    population.fv += 1

            # fv to b
            for i in range(population.fv):
                if random() < self.s.probs.vb:
                    population.fv -= 1
                    population.b += 1

    @staticmethod
    def transition_compartment(
        population_from: SubCompartmentPopulations,
        population_to: SubCompartmentPopulations,
        prob: float,
    ):
        # uv
        for i in range(population_from.uv):
            if random() < prob:
                population_from.uv -= 1
                population_to.uv += 1

        # fv
        for i in range(population_from.fv):
            if random() < prob:
                population_from.fv -= 1
                population_to.fv += 1

        # b
        for i in range(population_from.b):
            if random() < prob:
                population_from.b -= 1
                population_to.b += 1


class DivocEnv:
    def __init__(self):
        self.state = self.get_initial_state()

    def step(self, action: int):
        self.state = Transitions(self.state).step()
        self.state.step_count += 1
        self.state.is_done = self.state.step_count >= self.state.hyper_parameters.max_steps

        # Add noise to probs
        for prob_key in Probs.__fields__:
            noise = (random() - 0.5) * 0.05
            prob = getattr(self.state.probs, prob_key) + noise
            prob = max(0, prob)
            prob = min(1, prob)
            setattr(self.state.probs, prob_key, prob)

        # Add noise to epp
        self.state.epp += (random() - 0.5) * 0.01

    def reset(self):
        self.__init__()

    def get_state_dict(self):
        return self.state.dict()

    @staticmethod
    def get_initial_state():
        hp = SimHyperParams(
            action_durations={A.vaccine: 14, A.mask: 150, A.lockdown: 14, A.social_distancing: 30},
            max_steps=365,
            initial_population=1000,
        )
        state = State(
            populations=Populations(
                susceptible=SubCompartmentPopulations(uv=hp.initial_population),
                exposed1=SubCompartmentPopulations(),
                exposed2=SubCompartmentPopulations(),
                infected=SubCompartmentPopulations(),
                recovered=SubCompartmentPopulations(),
                hospitalized=SubCompartmentPopulations(),
                deceased=SubCompartmentPopulations(),
            ),
            probs=Probs(**{k: random() * 0.1 for k in Probs.__fields__}),
            action_residue={'vaccine': 1, 'mask': 0},
            epp=1.0,
            hyper_parameters=hp,
            action_mask={'noop': 1, 'vaccine': 0, 'mask': 1},
            step_count=0,
            is_done=False,
        )
        return state
