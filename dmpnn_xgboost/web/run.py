"""
Runs the web interface version of Chemprop.
This allows for training and predicting in a web browser.
"""

import os

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from app import app, db


class Args(Tap):
    host: str = '127.0.0.1'  # Host IP address
    port: int = 5000  # Port
    debug: bool = False  # Whether to run in debug mode
    demo: bool = False  # Display only demo features
    initdb: bool = False  # Initialize Database


if __name__ == "__main__":
    args = Args().parse_args()

    app.config['DEMO'] = args.demo

    db.init_app(app)

    if args.initdb or not os.path.isfile(app.config['DB_FILENAME']):
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    app.run(host=args.host, port=args.port, debug=args.debug)
