from src.app import create_app
from src.app.socket_handler import socketio

app = create_app()

if __name__ == '__main__':
    socketio.init_app(app)
    socketio.run(app, host="0.0.0.0", port=8080, debug=False, allow_unsafe_werkzeug=True)
