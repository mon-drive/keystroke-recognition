from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Configurazione (opzionale)
    # app.config['SECRET_KEY'] = 'tua_secret_key'

    # Registrare i blueprint (routes)
    from .routes import main
    app.register_blueprint(main)

    return app