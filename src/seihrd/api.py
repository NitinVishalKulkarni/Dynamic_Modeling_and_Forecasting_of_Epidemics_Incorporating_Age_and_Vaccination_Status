from fastapi import FastAPI, WebSocket
from seihrd.seihrd_env import SeihrdEnv


app = FastAPI()


@app.websocket_route("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print('accepted')
    env = SeihrdEnv()
    env.reset()
    await websocket.send_json({
        'type': 'next_state',
        'state': env.get_state_dict(),
        'error': None,
    })
    while True:
        data = await websocket.receive_json()
        action = data['action']
        env.step(action)
        await websocket.send_json({
            'type': 'next_state',
            'state': env.get_state_dict(),
            'error': None,
        })
