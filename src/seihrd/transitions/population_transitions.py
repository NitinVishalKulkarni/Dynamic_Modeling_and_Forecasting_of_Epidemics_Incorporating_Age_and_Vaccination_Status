from copy import deepcopy
from seihrd.base_models import SubCompPopulations, State


class PopulationTransitions:
    def __init__(self):
        pass

    def __call__(self, state: State):
        s = deepcopy(state)
        po = deepcopy(s.populations)
        pa = deepcopy(s.params)
        pa.noisy()

        i = (po.infected.total() ** pa.alpha) / po.total()

        s.populations.susceptible = SubCompPopulations(
            uv=int(
                po.susceptible.uv
                - (pa.beta * po.susceptible.uv * i)
                + (pa.e_s.uv * po.exposed.uv)
                - (pa.vfv * po.susceptible.uv)
            ),
            fv=int(
                po.susceptible.fv
                - (pa.beta * po.susceptible.fv * i)
                + (pa.e_s.fv * po.exposed.fv)
                + (pa.vfv * po.susceptible.uv)
                - (pa.vb * po.susceptible.fv)
            ),
            b=int(
                po.susceptible.b
                - (pa.beta * po.susceptible.b * i)
                + (pa.e_s.b * po.exposed.b)
                + (pa.vb * po.susceptible.fv)
            )
        )

        s.populations.exposed = SubCompPopulations(
            uv=int(
                po.exposed.uv
                + (pa.beta * po.susceptible.uv * i)
                + (pa.beta * po.recovered.uv * i)
                - (pa.e_i.uv * po.exposed.uv)
                - (pa.e2_i.uv * po.exposed.uv)
                - (pa.e_s.uv * po.exposed.uv)
                - (pa.e_r.uv * po.exposed.uv)
                - (pa.vfv * po.exposed.uv)
            ),
            fv=int(
                po.exposed.fv
                + (pa.beta * po.susceptible.fv * i)
                + (pa.beta * po.recovered.fv * i)
                - (pa.e_i.fv * po.exposed.fv)
                - (pa.e2_i.fv * po.exposed.fv)
                - (pa.e_s.fv * po.exposed.fv)
                - (pa.e_r.fv * po.exposed.fv)
                + (pa.vfv * po.exposed.uv)
                - (pa.vb * po.exposed.fv)
            ),
            b=int(
                po.exposed.b
                + (pa.beta * po.susceptible.b * i)
                + (pa.beta * po.recovered.b * i)
                - (pa.e_i.b * po.exposed.b)
                - (pa.e2_i.b * po.exposed.b)
                - (pa.e_s.b * po.exposed.b)
                - (pa.e_r.b * po.exposed.b)
                + (pa.vb * po.exposed.fv)
            )
        )

        s.populations.infected = SubCompPopulations(
            uv=self._updated_infected('uv', po, pa),
            fv=self._updated_infected('fv', po, pa),
            b=self._updated_infected('b', po, pa),
        )

        s.populations.hospitalized = SubCompPopulations(
            uv=self._updated_hospitalized('uv', po, pa),
            fv=self._updated_hospitalized('fv', po, pa),
            b=self._updated_hospitalized('b', po, pa),
        )

        s.populations.recovered = SubCompPopulations(
            uv=int(
                po.recovered.uv
                - (pa.beta * po.recovered.uv * i)
                + (pa.e_r.uv * po.exposed.uv)
                + (pa.i_r.uv * po.infected.uv)
                + (pa.h_r.uv * po.hospitalized.uv)
                - (pa.vfv * po.recovered.uv)
            ),
            fv=int(
                po.recovered.fv
                - (pa.beta * po.recovered.fv * i)
                + (pa.e_r.fv * po.exposed.fv)
                + (pa.i_r.fv * po.infected.fv)
                + (pa.h_r.fv * po.hospitalized.fv)
                + (pa.vfv * po.recovered.uv)
                - (pa.vb * po.recovered.fv)
            ),
            b=int(
                po.recovered.b
                - (pa.beta * po.recovered.b * i)
                + (pa.e_r.b * po.exposed.b)
                + (pa.i_r.b * po.infected.b)
                + (pa.h_r.b * po.hospitalized.b)
                + (pa.vb * po.recovered.fv)
            )
        )

        s.populations.deceased = SubCompPopulations(
            uv=self._updated_deceased('uv', po, pa),
            fv=self._updated_deceased('fv', po, pa),
            b=self._updated_deceased('b', po, pa),
        )

        return s

    @staticmethod
    def _updated_infected(comp, po, pa):
        return int(
            po.infected[comp]
            + (pa.e_i[comp] * po.exposed[comp])
            + (pa.e2_i[comp] * po.exposed[comp])
            - (pa.i_h[comp] * po.infected[comp])
            - (pa.i_r[comp] * po.infected[comp])
            - (pa.i_d[comp] * po.infected[comp])
            # AD: Why no vaccination conversions?
        )

    @staticmethod
    def _updated_hospitalized(comp, po, pa):
        return int(
            po.hospitalized[comp]
            + (pa.i_h[comp] * po.infected[comp])
            - (pa.h_r[comp] * po.hospitalized[comp])
            - (pa.h_d[comp] * po.hospitalized[comp])
            # AD: Why no vaccination conversions?
        )

    @staticmethod
    def _updated_deceased(comp, po, pa):
        return int(
            po.deceased[comp]
            + (pa.i_d[comp] * po.infected[comp])
            + (pa.h_d[comp] * po.hospitalized[comp])
        )
