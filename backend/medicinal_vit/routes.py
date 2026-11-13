"""Flask routes for the independent medicinal ViT service."""

from flask import Blueprint, current_app, jsonify, request

from .service import ImageInputError, MedicinalViTService, ModelInferenceError


def create_medicinal_blueprint(service: MedicinalViTService):
    blueprint = Blueprint("medicinal_vit", __name__)

    @blueprint.post("/medicinal/predict")
    def predict():
        uploaded_file = request.files.get("file")
        include_all_value = request.args.get(
            "include_all",
            request.form.get("include_all", "false"),
        ).strip().lower()
        if include_all_value not in {"true", "false", "1", "0"}:
            return jsonify({"error": "include_all must be true or false."}), 400

        try:
            result = service.predict_upload(
                uploaded_file,
                include_all=include_all_value in {"true", "1"},
            )
        except ImageInputError as error:
            return jsonify({"error": str(error)}), 400
        except ModelInferenceError as error:
            current_app.logger.exception("Medicinal ViT inference failed")
            return jsonify({"error": str(error)}), 500
        except Exception:
            current_app.logger.exception("Unexpected medicinal ViT prediction failure")
            return jsonify({"error": "Prediction failed due to an internal model error."}), 500

        return jsonify(result)

    @blueprint.get("/medicinal/health")
    def health():
        return jsonify(service.health())

    return blueprint
