from fastapi import FastAPI, WebSocket
from base_models import a2i
from divoc_env import DivocEnv


app = FastAPI()


@app.websocket_route("/ws/divoc")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print('accepted')
    env = DivocEnv()
    env.reset()
    await websocket.send_json({
        'type': 'next_state',
        'state': env.get_state_dict(),
        'error': None,
    })
    while True:
        data = await websocket.receive_json()
        action = a2i[data['action']]
        env.step(action)
        await websocket.send_json({
            'type': 'next_state',
            'state': env.get_state_dict(),
            'error': None,
        })
