class Divoc {
    constructor() {
        this.state = default_state
        console.log('Trying to connect')
        this.ws = new WebSocket(consts.websocket_url)
        this.sim = new SimUI(this)
        this.stats = new Stats(this)

        this.ws.onmessage = event => this.received(JSON.parse(event.data))
    }

    received({state}) {
        this.state = state
        this.sim.draw()
        this.stats.step()
    }

    step() {
        if (this.ws?.readyState === WebSocket.OPEN){
            this.ws.send(JSON.stringify({
                type: 'step',
                action: index_to_actions[0],
                // prev_state: this.state,
            }))
        }
        else {
            console.log('Not connected')
        }
    }
}


$(document).ready(function () {
    let env = new Divoc()
    setInterval(() => env.step(), consts.step_interval);
});

