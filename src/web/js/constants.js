consts = {
    websocket_url: "ws://localhost:8000/ws",
    step_interval: 1000,
}

Actions = {
    noop: 'noop',
    social_distancing: 'social_distancing',
    vaccine: 'vaccine',
    mask: 'mask',
    lockdown: 'lockdown',
}
index_to_actions = [
    Actions.social_distancing,
    Actions.lockdown,
    Actions.mask,
    Actions.vaccine,
]
action_to_index = index_to_actions.map((a, index) => index)

_sub_compartment_populations = {uv: 0, fv: 0, b: 0}
_sub_compartment_preds = {vfv: 0.0, vb: 0.0}

default_state = {
    populations: {
        susceptible: _sub_compartment_populations,
        exposed: _sub_compartment_populations,
        infected: _sub_compartment_populations,
        recovered: _sub_compartment_populations,
        hospitalized: _sub_compartment_populations,
        deceased: _sub_compartment_populations,
    },
    params: {
        vfv: 0.0,
        vb: 0.0,
        alpha: 0.0,
        beta: 0.0,
        e_s: 0.0,
        e_i: 0.0,
        i_r: 0.0,
        i_h: 0.0,
        i_d: 0.0,
        e2_i: 0.0,
        h_r: 0.0,
        h_d: 0.0,
        e_r: 0.0,
    },
    action_in_effect: {
        social_distancing: 0,
        lockdown: 0,
        vaccine: 0,
        mask: 0,
    },
    epp: 1.0,
    hyper_params: {
        action_durations: {  // in days
            social_distancing: 100,
            lockdown: 10,
            vaccine: 50,
            mask: 150,
        },
        action_cool_downs: {
            social_distancing: 30,
            lockdown: 10,
            vaccine: 100,
            mask: 30,
        },
        max_steps: 365,
        initial_population: 1000,
    },
    action_mask: [1, 1, 1, 1],
    step_count: 0,
    is_done: false,
}

default_send_message = {
    type: 'step',
    action: index_to_actions[0],
    prev_state: default_state,
}

default_receive_message = {
    type: 'next_state',
    state: default_state,
    error: null,
}
