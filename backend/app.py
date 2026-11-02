import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# PATHS
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "ML" / "models" / "leaflens_mobilenetv2_grouped_windows.keras"
LABELS_PATH = ROOT / "ML" / "models" / "class_names_grouped_windows.json"
TEMPLATE_DIR = ROOT / "frontend" / "templates"

IMAGE_SIZE = (160, 160)
MAX_FILE_SIZE = 10 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 60.0

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# React/Vite development server runs on port 8080.
CORS(app, resources={r"/predict": {"origins": [
    "http://localhost:8080", "http://127.0.0.1:8080"
]}})


# PLANT INFORMATION
# Unlisted species will show unknown edibility.

def plant_info(uses, edibility, safety):
    return {
        "uses": uses,
        "edibility": edibility,
        "safety": safety
    }


PLANT_INFO = {
    "Aloevera": plant_info(
        "Aloe gel is used in topical skincare products.",
        "Do not eat based on this prediction. Aloe latex can cause harmful effects.",
        "Verify the species and plant part. Avoid homemade preparations for ingestion."
    ),
    "Amla": plant_info(
        "Amla fruit is used in foods and traditional preparations.",
        "The fruit is commonly eaten when correctly identified.",
        "Do not assume every part or preparation is safe."
    ),
    "Amruthaballi": plant_info(
        "Tinospora cordifolia is a possible identification; it is traditionally used in herbal preparations.",
        "Do not consume based on this prediction. Herbal preparations may cause adverse effects.",
        "Confirm botanical identity and discuss medicinal use with a qualified professional."
    ),
    "Arali": plant_info(
        "Arali may refer to an ornamental oleander plant in some regional usage.",
        "POTENTIALLY FATAL TOXICITY. Do not eat any part if this is oleander.",
        "Oleander can cause dangerous heart effects. Keep away from children and pets."
    ),
    "Astma_weed": plant_info(
        "May refer to a plant used in traditional respiratory remedies; exact identity needs verification.",
        "Unknown. Do not consume.",
        "The common name is insufficient to establish identity or medicinal safety."
    ),
    "Badipala": plant_info(
        "Uses require confirmation of the botanical species.",
        "Unknown. Do not consume until identified.",
        "Verify the regional name with a botanist or agricultural expert."
    ),
    "Balloon_Vine": plant_info(
        "Cardiospermum halicacabum is used ornamentally and in some traditional preparations.",
        "Do not eat based on this prediction; edibility needs species-specific verification.",
        "Traditional use does not establish safety or effectiveness."
    ),
    "Bamboo": plant_info(
        "Bamboo is used in construction, crafts, furniture and paper production.",
        "Young shoots of certain species are eaten after appropriate preparation. This prediction does not establish species identity or safe preparation.",
        "Never eat an unidentified bamboo plant or raw shoots without species-specific guidance."
    ),
    "Beans": plant_info(
        "Bean plants are grown for food and agriculture.",
        "Some bean varieties are edible when properly prepared; some raw beans contain harmful compounds.",
        "Confirm the species and preparation requirements before eating."
    ),
    "Betel": plant_info(
        "Betel leaves are used in some cultural practices and preparations.",
        "Do not assume the leaf or any preparation is safe to eat. Chewing mixtures may have serious health risks.",
        "Avoid tobacco-containing betel mixtures; confirm plant identity."
    ),
    "Bhrami": plant_info(
        "May refer to Bacopa monnieri or another plant called Brahmi regionally.",
        "Do not consume based on this prediction.",
        "Brahmi is an ambiguous common name. Confirm the botanical species before medicinal use."
    ),
    "Bringaraja": plant_info(
        "Eclipta prostrata is used in some traditional hair and herbal preparations.",
        "Edibility is not verified for this identification.",
        "Do not ingest herbal preparations without qualified guidance."
    ),
    "Caricature": plant_info(
        "The dataset label needs botanical identification before uses can be assigned.",
        "Unknown. Do not consume.",
        "Confirm the scientific name; do not infer safety from this label."
    ),
    "Castor": plant_info(
        "Castor oil is produced from processed seeds for specific industrial and regulated applications.",
        "TOXIC. Castor seeds contain ricin, a highly toxic substance. Never eat them.",
        "Keep seeds away from children and pets. Seek urgent medical assistance if ingestion is suspected."
    ),
    "Catharanthus": plant_info(
        "Catharanthus roseus is grown as an ornamental plant and is a source of compounds used in regulated medicines.",
        "TOXICITY RISK. Do not eat or prepare home remedies from this plant.",
        "Contains potent alkaloids. Medicinal compounds must only be used in regulated treatment."
    ),
    "Chakte": plant_info(
        "Regional plant identity and uses require verification.",
        "Unknown. Do not consume.",
        "Obtain a botanical identification before assigning uses or safety."
    ),
    "Chilly": plant_info(
        "Chilli plants produce fruits used as spices and vegetables.",
        "Correctly identified chilli fruits are commonly eaten; leaves and other parts should not be assumed edible.",
        "Chilli can irritate skin and eyes. Confirm the plant before consumption."
    ),
    "Citron lime (herelikai)": plant_info(
        "Citrus fruits may be used in cooking, drinks and flavouring; this regional label needs confirmation.",
        "Only confirmed edible citrus fruit should be eaten.",
        "Verify the exact citrus species and fruit identity."
    ),
    "Coffee": plant_info(
        "Coffee plants are cultivated for their beans, which are processed into coffee.",
        "Processed coffee beans are commonly consumed. Do not eat unidentified berries or plant parts.",
        "Caffeine can cause adverse effects in excess; avoid unverified plant preparations."
    ),
    "Common rue(naagdalli)": plant_info(
        "Rue is grown as an ornamental plant and has a history of traditional use.",
        "UNSAFE FOR SELF-MEDICATION. Rue can be toxic, particularly in concentrated preparations.",
        "May cause photosensitivity and serious adverse effects. Avoid ingestion, especially during pregnancy."
    ),
    "Coriender": plant_info(
        "Coriander leaves are widely used as a culinary herb.",
        "Coriander leaves are commonly eaten when correctly identified.",
        "Confirm identity before eating an unfamiliar plant."
    ),
    "Curry": plant_info(
        "Curry leaf is commonly used to flavour foods.",
        "Curry leaves are commonly eaten when correctly identified.",
        "Verify identity before eating an unfamiliar plant."
    ),
    "Doddpathre": plant_info(
        "May refer to an aromatic Coleus/Plectranthus species used in regional cooking and traditional practices.",
        "Do not consume based only on this prediction; regional names may refer to different plants.",
        "Confirm the botanical species and appropriate use."
    ),
    "Drumstick": plant_info(
        "Moringa is cultivated for its pods and leaves and is used as food.",
        "Certain properly identified moringa parts are eaten as food; preparation varies by part.",
        "Confirm the plant and plant part before eating."
    ),
    "Ekka": plant_info(
        "May refer to Calotropis, a plant used in some traditional practices.",
        "TOXIC. Do not eat any part.",
        "Calotropis sap can irritate skin and eyes and can be poisonous. Avoid home remedies."
    ),
    "Eucalyptus": plant_info(
        "Eucalyptus is used for timber, essential oils and other commercial products.",
        "Do not eat the leaves or essential oil. Eucalyptus oil can be poisonous if swallowed.",
        "Keep essential oils away from children. Use only products intended for their labelled purpose."
    ),
    "Ganigale": plant_info(
        "Uses cannot be assigned reliably until the local name is matched to a botanical species.",
        "Unknown. Do not consume.",
        "Verify the species with a botanist or agricultural expert."
    ),
    "Ganike": plant_info(
        "The regional name needs botanical verification before uses can be described.",
        "Unknown. Do not consume.",
        "Do not assume edibility or medicinal safety from the model label."
    ),
    "Gasagase": plant_info(
        "May refer regionally to poppy seeds or a related plant; the leaf label alone is insufficient to identify it.",
        "Do not consume based on this prediction. Food uses depend on confirmed identity and plant part.",
        "Confirm the species and local name; do not use unidentified plants medicinally."
    ),
    "Ginger": plant_info(
        "Ginger rhizomes are used as a culinary spice and in traditional preparations.",
        "The familiar culinary rhizome is commonly eaten when correctly identified.",
        "The model identifies a leaf image, not the identity or safety of a harvested rhizome."
    ),
    "Globe Amarnath": plant_info(
        "Grown as an ornamental plant; some related amaranths are used as food.",
        "Edibility of this specific species is not confirmed by this label.",
        "Do not assume related species share the same food safety."
    ),
    "Guava": plant_info(
        "Guava is grown for its edible fruit; leaves are also used in some traditional preparations.",
        "The correctly identified ripe fruit is commonly eaten.",
        "This leaf prediction does not confirm fruit identity or medicinal safety."
    ),
    "Henna": plant_info(
        "Henna leaves are used to prepare dye for skin and hair.",
        "Do not eat henna leaves or ingest henna products.",
        "Henna is for external dye use; avoid ingestion and be cautious with products of uncertain composition."
    ),
    "Hibiscus": plant_info(
        "Hibiscus species are grown ornamentally; certain species and parts are used in foods and drinks.",
        "Edibility depends on the exact species and plant part; this prediction does not confirm either.",
        "Verify the species before consuming any part."
    ),
    "Honge": plant_info(
        "May refer to Pongamia pinnata, a tree used for shade and oil-related applications.",
        "Do not eat seeds or other parts based on this prediction.",
        "Confirm the species and avoid ingesting unprocessed plant products."
    ),
    "Insulin": plant_info(
        "May refer to a plant marketed as an insulin plant, often Costus igneus, in traditional practices.",
        "Do not consume based on this prediction or use it to manage diabetes.",
        "Plant identity and medicinal effectiveness require verification. Never replace prescribed diabetes treatment."
    ),
    "Jackfruit": plant_info(
        "Jackfruit is cultivated for its fruit and seeds, which are used in food after appropriate preparation.",
        "Ripe fruit and properly cooked seeds are commonly eaten when correctly identified.",
        "The leaf prediction does not establish fruit identity or safe preparation."
    ),
    "Jasmine": plant_info(
        "Jasmine species are grown for fragrant flowers, ornamentals and perfumes.",
        "Do not assume the leaves or flowers are edible; species and preparation matter.",
        "Confirm the exact species before ingestion or medicinal use."
    ),
    "Kambajala": plant_info(
        "This regional label requires botanical verification.",
        "Unknown. Do not consume.",
        "A reliable scientific name is needed before describing uses or edibility."
    ),
    "Kasambruga": plant_info(
        "The local name cannot be confidently mapped to a botanical species here.",
        "Unknown. Do not consume.",
        "Verify the scientific name with a botanical expert."
    ),
    "Kohlrabi": plant_info(
        "Kohlrabi is grown as a vegetable for its swollen stem and leaves.",
        "Correctly identified kohlrabi is commonly eaten as food.",
        "Wash produce and confirm identity before eating."
    ),
    "Lantana": plant_info(
        "Lantana is primarily an ornamental plant and can be invasive.",
        "TOXICITY RISK. Do not eat berries, leaves or other parts.",
        "Some lantana parts can poison people and animals. Keep away from children and pets."
    ),
    "Lemon": plant_info(
        "Lemon fruit is used in cooking, drinks and flavouring.",
        "The correctly identified fruit is commonly eaten as food.",
        "The leaf prediction does not confirm the identity of a fruit or its safety."
    ),
    "Malabar_Nut": plant_info(
        "May refer to Justicia adhatoda, traditionally used in respiratory herbal preparations.",
        "Do not consume or self-medicate based on this prediction.",
        "Verify the species and seek qualified medical advice before medicinal use."
    ),
    "Malabar_Spinach": plant_info(
        "Malabar spinach is cultivated as a leafy vegetable.",
        "Correctly identified leaves and shoots are commonly eaten after appropriate washing and preparation.",
        "Confirm identity before eating an unfamiliar plant."
    ),
    "Mango": plant_info(
        "Mango is cultivated for its edible fruit; leaves have cultural and traditional uses.",
        "Ripe mango fruit is commonly eaten. Do not eat leaves based on this prediction.",
        "Mango sap and peel can irritate or trigger reactions in some people."
    ),
    "Marigold": plant_info(
        "Marigolds are grown as ornamental plants; some species have horticultural uses.",
        "Do not assume this species or its parts are edible.",
        "Confirm the species before using flowers or leaves in food or remedies."
    ),
    "Mint": plant_info(
        "Some mint species are used in food, drinks and flavouring.",
        "Some correctly identified mint leaves are edible; this prediction may be wrong.",
        "Confirm the species before eating."
    ),
    "Neem": plant_info(
        "Neem has traditional topical and agricultural uses.",
        "Do not ingest neem oil or use this prediction to decide edibility.",
        "Some neem preparations can be harmful if swallowed. Seek professional guidance before medicinal use."
    ),
    "Nelavembu": plant_info(
        "May refer to Andrographis paniculata, used in some traditional herbal preparations.",
        "Do not consume based on this prediction or use it as self-treatment.",
        "Verify the species and discuss potential interactions with a qualified healthcare professional."
    ),
    "Nerale": plant_info(
        "May refer to jamun (Syzygium cumini), a tree cultivated for its fruit.",
        "The correctly identified ripe fruit is commonly eaten; this label does not confirm identity.",
        "Confirm the species and plant part before consumption."
    ),
    "Nooni": plant_info(
        "May refer to noni (Morinda citrifolia), whose fruit is used in some food and traditional products.",
        "Do not consume based on this prediction. Safety depends on the exact species and preparation.",
        "Verify the plant and seek professional advice before using concentrated products."
    ),
    "Onion": plant_info(
        "Onion is cultivated as a vegetable and culinary flavouring.",
        "Correctly identified onion bulbs are commonly eaten.",
        "The leaf prediction does not confirm that a harvested bulb is onion."
    ),
    "Padri": plant_info(
        "This regional plant name requires botanical verification.",
        "Unknown. Do not consume.",
        "Confirm the scientific name before assigning food or medicinal uses."
    ),
    "Palak(Spinach)": plant_info(
        "Spinach is cultivated as a leafy vegetable.",
        "Correctly identified spinach leaves are commonly eaten after washing and appropriate preparation.",
        "People prone to kidney stones may need individualized advice about high-oxalate foods."
    ),
    "Papaya": plant_info(
        "Papaya is cultivated for its fruit; other parts have traditional uses.",
        "Ripe fruit is commonly eaten when correctly identified. Do not eat leaves or seeds based on this prediction.",
        "Papaya latex can cause reactions; extra caution is needed during pregnancy and with unverified preparations."
    ),
    "Parijatha": plant_info(
        "May refer to Nyctanthes arbor-tristis, grown for its fragrant flowers.",
        "Do not consume based on this prediction; food safety is not established here.",
        "Verify the botanical species before medicinal use."
    ),
    "Pea": plant_info(
        "Pea plants are grown for their edible seeds and pods.",
        "Correctly identified peas are commonly eaten; preparation depends on the variety.",
        "Confirm identity before eating an unfamiliar plant."
    ),
    "Pomoegranate": plant_info(
        "Pomegranate is cultivated for its edible arils and juice.",
        "The correctly identified ripe fruit is commonly eaten.",
        "This leaf prediction does not confirm the identity of a harvested fruit."
    ),
    "Pumpkin": plant_info(
        "Pumpkin plants are grown for their fruits and, in some varieties, edible seeds and flowers.",
        "Edibility depends on the variety and plant part. Do not eat an unknown or unusually bitter gourd-like fruit.",
        "Some cucurbits can contain harmful bitter compounds. Do not consume unusually bitter fruits."
    ),
    "Raddish": plant_info(
        "Radish is grown for its edible root and, in some varieties, leaves.",
        "Correctly identified radish roots are commonly eaten.",
        "Confirm identity and wash produce before eating."
    ),
    "Rose": plant_info(
        "Roses are grown ornamentally and used in some food, fragrance and cosmetic products.",
        "Only correctly identified, appropriately grown rose petals are used in some foods. Do not assume leaves are edible.",
        "Ornamental plants may have pesticide residues. Confirm species and growing conditions."
    ),
    "Sampige": plant_info(
        "May refer to a champak tree, valued for fragrant flowers.",
        "Do not assume leaves, flowers or other parts are edible.",
        "Confirm the botanical species before ingestion or medicinal use."
    ),
    "Sapota": plant_info(
        "Sapota (chikoo) is cultivated for its sweet fruit.",
        "The correctly identified ripe fruit is commonly eaten.",
        "Do not consume seeds or other parts based on this prediction."
    ),
    "Seethaashoka": plant_info(
        "May refer to Ashoka-related trees used ornamentally and in traditional practices.",
        "Unknown for this identification. Do not consume.",
        "Common names can refer to different trees. Confirm the scientific name."
    ),
    "Seethapala": plant_info(
        "May refer to custard apple (Annona squamosa), grown for its fruit.",
        "The correctly identified ripe pulp is commonly eaten; seeds must not be eaten.",
        "Custard apple seeds contain biologically active compounds and should not be ingested."
    ),
    "Spinach1": plant_info(
        "The dataset label does not identify the exact spinach species.",
        "Unknown until the species is confirmed. Do not consume based only on this prediction.",
        "Confirm the botanical identity and food preparation requirements."
    ),
    "Tamarind": plant_info(
        "Tamarind is cultivated for its sour-sweet pulp used in food.",
        "The correctly identified ripe pulp is commonly eaten.",
        "Do not assume leaves, seeds or other parts are interchangeable with edible pulp."
    ),
    "Taro": plant_info(
        "Taro is cultivated for its starchy corms and some edible leaves.",
        "Only correctly identified parts prepared appropriately are eaten. Raw taro can cause intense irritation.",
        "Do not eat raw taro. Proper preparation is essential."
    ),
    "Tecoma": plant_info(
        "Tecoma species are commonly grown as ornamental flowering plants.",
        "Edibility is not verified for this identification. Do not consume.",
        "Confirm the species before considering any food or medicinal use."
    ),
    "Thumbe": plant_info(
        "May refer to Leucas species used in some regional traditional practices.",
        "Unknown for this identification. Do not consume based on this prediction.",
        "Confirm the scientific name and seek qualified advice before medicinal use."
    ),
    "Tomato": plant_info(
        "Tomato plants are grown for their edible fruits.",
        "Ripe tomato fruits are commonly eaten. Leaves and stems should not be eaten as food.",
        "Green plant parts contain glycoalkaloids and may cause illness if ingested."
    ),
    "Tulsi": plant_info(
        "Tulsi (holy basil) is used as a culinary herb and in traditional wellness practices.",
        "Commonly used as a culinary herb when correctly identified; this result does not confirm identity.",
        "Do not use it as a replacement for prescribed treatment."
    ),
    "Turmeric": plant_info(
        "Turmeric rhizomes are used as a culinary spice and colouring ingredient.",
        "The correctly identified, prepared rhizome is commonly eaten.",
        "The leaf prediction does not confirm the identity or safety of a harvested rhizome."
    ),
    "ashoka": plant_info(
        "The label may refer to an Ashoka tree, but exact species identification is needed.",
        "Unknown. Do not consume based on this prediction.",
        "Do not confuse true Ashoka with similarly named ornamental trees."
    ),
    "camphor": plant_info(
        "Camphor is used in some topical products, fragrances and cultural practices.",
        "DO NOT EAT camphor or ingest camphor products; poisoning can be severe.",
        "Keep camphor away from children. Swallowing it requires urgent medical advice."
    ),
    "kamakasturi": plant_info(
        "May refer to an aromatic basil-like plant used in regional food or traditional practices; identity needs verification.",
        "Do not consume until the exact species is confirmed.",
        "Verify the scientific name and distinguish it from similar aromatic plants."
    ),
    "kepala": plant_info(
        "This regional name needs a confirmed botanical identity before uses can be assigned.",
        "Unknown. Do not consume.",
        "Ask a botanist or agricultural expert to verify the plant."
    )
}


DEFAULT_PLANT_INFO = {
    "uses": "Information is not available for this label.",
    "edibility": "UNKNOWN. Do not eat or taste this plant based on an AI prediction.",
    "safety": "Confirm the species with a reliable botanical source or qualified expert."
}



# LOAD MODEL
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not LABELS_PATH.is_file():
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with LABELS_PATH.open("r", encoding="utf-8") as file:
    class_names = json.load(file)

if model.output_shape[-1] != len(class_names):
    raise ValueError("Model output count does not match class labels.")

print(f"Model loaded: {MODEL_PATH.name}")
print(f"Plant classes: {len(class_names)}")


# WEBSITE
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/app")
def web_app():
    return render_template("index.html")


# HEALTH CHECK
@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "app": "LeafLens AI",
        "classes": len(class_names)
    })


# PREDICTION API
@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({"error": "Please select an image."}), 400

    try:
        with Image.open(uploaded_file.stream) as original:
            # Convert to RGB; do not normalize because the model includes
            # MobileNetV2 preprocessing internally.
            image = original.convert("RGB")
            image_array = np.asarray(image, dtype=np.float32)

        image_array = tf.image.resize(
            image_array, IMAGE_SIZE, method="bilinear"
        )
        image_array = np.expand_dims(image_array.numpy(), axis=0)

        # Preprocessing is already inside the trained model.
        probabilities = model.predict(image_array, verbose=0)[0]

        # Return only the highest-scoring prediction.
        predicted_index = int(np.argmax(probabilities))
        species = str(class_names[predicted_index])
        confidence = float(probabilities[predicted_index]) * 100

        low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        # Match plant information to the predicted label.
        info = PLANT_INFO.get(species, DEFAULT_PLANT_INFO)

        warning = (
            "Low-confidence result. The plant identity is uncertain."
            if low_confidence else
            "AI prediction only. The model score does not prove "
            "plant identity, edibility, or medicinal safety."
        )

        return jsonify({
            "prediction": species,
            "confidence": round(confidence, 2),
            "low_confidence": low_confidence,
            "warning": warning,
            "plant_info": info
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "Invalid image. Upload a valid JPG, PNG, or BMP."
        }), 400


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "Image exceeds the 10 MB upload limit."
    }), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({
        "error": "Page not found. Open / or /app."
    }), 404


if __name__ == "__main__":
    print("LeafLens AI starting...")
    print("Flask API: http://127.0.0.1:8000/predict")
    print("Health check: http://127.0.0.1:8000/health")
    print("React UI: http://localhost:8080/")
    app.run(host="127.0.0.1", port=8000, debug=False)
