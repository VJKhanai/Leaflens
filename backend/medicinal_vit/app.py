"""Standalone Flask entry point for medicinal plant ViT inference."""

from flask import Flask, jsonify

from .routes import create_medicinal_blueprint
from .service import MedicinalViTService


def create_app(service=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    medicinal_service = service if service is not None else MedicinalViTService()
    app.register_blueprint(create_medicinal_blueprint(medicinal_service))

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": "Image exceeds the 10 MB upload limit."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
