from flask_socketio import SocketIO

from src.app.resources import jeopardy, dbpedia

socketio = SocketIO(cors_allowed_origins="*")


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    no_jeopardy_questions = jeopardy.aggregate.over_all(total_count=True).total_count
    no_articles = dbpedia.aggregate.over_all(total_count=True).total_count
    socketio.emit('initial_data', {
        'total_jeopardy_questions': no_jeopardy_questions,
        'total_articles': no_articles
    })
