import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# PATHS
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "ML" / "models" / "leaflens_mobilenetv2_hybrid_finetuned.keras"

LABELS_PATH = ROOT / "ML" / "models" / "class_names_hybrid.json"
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


# =========================================================
# PLANT INFORMATION DATABASE
# =========================================================

def plant_info(
    common_name,
    scientific_name,
    description,
    benefits,
    disadvantages,
    proper_usage,
    applications,
    edibility,
    harmful_compounds,
    if_consumed,
    precautions
):
    return {
        "common_name": common_name,
        "scientific_name": scientific_name,
        "description": description,
        "benefits": benefits,
        "disadvantages": disadvantages,
        "proper_usage": proper_usage,
        "applications": applications,
        "edibility": edibility,
        "harmful_compounds": harmful_compounds,
        "if_consumed": if_consumed,
        "precautions": precautions,

        # Backward compatibility
        "uses": " ".join(applications),
        "safety": precautions
    }



PLANT_INFO = {

    "Aloevera": plant_info(
        "Aloe Vera",
        "Aloe vera",
        "A succulent plant with thick, fleshy leaves containing clear gel.",
        ["Aloe gel is used in moisturizing skincare products."],
        ["May cause skin irritation or allergic reactions.",
         "Aloe latex can cause harmful effects if swallowed."],
        "Use commercially formulated topical products according to their labels.",
        ["Skincare", "Cosmetics", "Personal care"],
        "Do not consume based on AI identification.",
        ["Aloin and other anthraquinones in aloe latex"],
        "Swallowing aloe latex may cause abdominal cramps and diarrhea.",
        "Avoid oral aloe latex. Verify the plant and plant part."
    ),

    "Amla": plant_info(
        "Indian Gooseberry",
        "Phyllanthus emblica",
        "A tree producing round, sour fruits.",
        ["Fruit provides vitamin C.",
         "Used in food and traditional preparations."],
        ["Excessive intake may cause digestive discomfort."],
        "Use properly identified fruit in washed food preparations.",
        ["Food", "Juices", "Pickles"],
        "Ripe fruit is commonly eaten.",
        ["No specific toxic compound assigned to ordinary fruit consumption."],
        "Excessive intake may cause digestive discomfort.",
        "Do not assume every plant part or preparation is safe."
    ),

    "Amruthaballi": plant_info(
        "Amruthaballi / Guduchi",
        "Possibly Tinospora cordifolia",
        "A climbing plant traditionally used in herbal preparations.",
        ["Has a history of traditional herbal use."],
        ["Some Guduchi preparations have been associated with liver injury."],
        "Do not self-medicate. Confirm the species before medicinal use.",
        ["Traditional herbal preparations"],
        "Do not consume based on AI identification.",
        ["Plant constituents vary by species and preparation."],
        "Some preparations may cause adverse effects, including liver injury.",
        "Seek qualified medical advice before medicinal use."
    ),

    "Arali": plant_info(
        "Oleander (possible identification)",
        "Possibly Nerium oleander",
        "An ornamental shrub with flowers and narrow leaves.",
        ["Primarily used as an ornamental plant."],
        ["Can cause potentially fatal poisoning."],
        "Ornamental use only. Never prepare food or home remedies from it.",
        ["Ornamental gardening"],
        "TOXIC. Do not eat any part if this is oleander.",
        ["Cardiac glycosides, including oleandrin"],
        "May cause vomiting and dangerous heart rhythm disturbances.",
        "Keep away from children and pets. Seek emergency help if ingestion is suspected."
    ),

    "Astma_weed": plant_info(
        "Asthma Weed",
        "Scientific identity unconfirmed",
        "Dataset label associated with a regional plant name.",
        ["Traditional uses are reported but the species is unconfirmed."],
        ["Incorrect identification may result in unsafe use."],
        "Do not use medicinally until identified by a botanical expert.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Effects cannot be predicted without species identification.",
        "Obtain a verified scientific name before assigning uses or safety."
    ),

    "Badipala": plant_info(
        "Badipala",
        "Scientific identity unconfirmed",
        "Regional plant name requiring botanical verification.",
        ["No verified species-specific benefits available."],
        ["Incorrect identification may lead to unsafe use."],
        "No recommended use until the species is confirmed.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Effects cannot be determined without identification.",
        "Confirm the scientific name with a botanical expert."
    ),

    "Balloon_Vine": plant_info(
        "Balloon Vine",
        "Cardiospermum halicacabum",
        "A climbing herb with inflated, balloon-like seed capsules.",
        ["Has traditional topical and herbal uses."],
        ["May cause allergic reactions or irritation."],
        "Do not ingest based on AI identification.",
        ["Traditional preparations", "Ornamental gardening"],
        "Edibility not established.",
        ["Species-specific composition requires verification."],
        "Ingestion effects are not reliably established.",
        "Confirm the species before medicinal use."
    ),

    "Bamboo": plant_info(
        "Bamboo",
        "Bambusoideae",
        "A group of fast-growing grasses with hollow or solid stems.",
        ["Used in construction, furniture and paper production.",
         "Some young shoots are used as food."],
        ["Some raw shoots contain cyanogenic compounds."],
        "Only verified edible species and properly processed shoots should be eaten.",
        ["Construction", "Furniture", "Paper", "Food"],
        "Some young shoots are edible after proper processing.",
        ["Cyanogenic glycosides in some bamboo shoots"],
        "Improperly prepared shoots may cause poisoning.",
        "Never eat raw or unidentified bamboo shoots."
    ),

    "Beans": plant_info(
        "Bean Plant",
        "Fabaceae",
        "A group of legume plants cultivated for pods and seeds.",
        ["Provides plant protein and dietary fibre in edible varieties."],
        ["Some raw beans contain compounds that can cause illness."],
        "Eat only positively identified edible varieties prepared properly.",
        ["Food", "Agriculture"],
        "Variety-dependent. Not every bean is safe raw.",
        ["Phytohaemagglutinin in some beans"],
        "Raw or improperly cooked beans may cause vomiting and diarrhea.",
        "Identify the variety and follow proper cooking instructions."
    ),

    "Betel": plant_info(
        "Betel Leaf",
        "Piper betle",
        "A climbing plant with broad aromatic leaves.",
        ["Used in cultural practices and traditional preparations."],
        ["Betel quid mixtures, especially those containing areca nut and tobacco, carry serious health risks."],
        "Avoid tobacco-containing betel mixtures.",
        ["Cultural practices", "Traditional preparations"],
        "Do not assume medicinal preparations or chewing mixtures are safe.",
        ["Risks depend on the preparation; areca nut and tobacco are major concerns."],
        "Chewing betel quid can increase oral disease and cancer risk.",
        "Avoid tobacco and areca nut mixtures."
    ),
    
    "Bhrami": plant_info(
        "Brahmi", "Possibly Bacopa monnieri",
        "A regional name that can refer to different plants.",
        ["Some species called Brahmi have traditional herbal uses."],
        ["May cause digestive discomfort or medicine interactions."],
        "Confirm the botanical species before medicinal use.",
        ["Traditional herbal preparations"],
        "Unknown. Do not consume based on prediction.",
        ["Species-dependent compounds"],
        "Possible digestive or other adverse effects.",
        "Verify the exact scientific name."
    ),

    "Bringaraja": plant_info(
        "Bhringraj", "Eclipta prostrata",
        "A small herb traditionally used in hair and herbal preparations.",
        ["Used in some hair-care products."],
        ["Evidence for many medicinal claims is limited."],
        "Use verified external products according to their labels.",
        ["Hair care", "Traditional preparations"],
        "Edibility not established.",
        ["Composition depends on preparation."],
        "Ingestion safety is not established.",
        "Avoid self-medication."
    ),

    "Caricature": plant_info(
        "Caricature Plant", "Scientific identity unconfirmed",
        "Dataset label requiring botanical identification.",
        ["No verified benefits assigned."],
        ["Incorrect identification can result in unsafe use."],
        "No food or medicinal use until identified.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Effects cannot be predicted.",
        "Confirm the scientific name."
    ),

    "Castor": plant_info(
        "Castor", "Ricinus communis",
        "A shrub with large palmate leaves and seed capsules.",
        ["Processed castor oil has industrial applications."],
        ["Seeds are highly poisonous."],
        "Never process or consume seeds at home.",
        ["Industrial oil", "Manufacturing"],
        "TOXIC. Never eat castor seeds.",
        ["Ricin in castor seeds"],
        "Can cause severe vomiting, diarrhea and life-threatening poisoning.",
        "Keep seeds away from children and pets. Seek urgent medical care after suspected ingestion."
    ),

    "Catharanthus": plant_info(
        "Madagascar Periwinkle", "Catharanthus roseus",
        "An ornamental flowering plant with glossy leaves.",
        ["Source of compounds used in regulated cancer medicines."],
        ["Contains potent alkaloids and can be poisonous."],
        "Ornamental use only. Never prepare homemade medicines.",
        ["Ornamental gardening", "Pharmaceutical research"],
        "TOXICITY RISK. Do not eat any part.",
        ["Vincristine, vinblastine and other alkaloids"],
        "Ingestion may cause harmful effects.",
        "Keep away from children and pets."
    ),

    "Chakte": plant_info(
        "Chakte", "Scientific identity unconfirmed",
        "Regional plant name requiring botanical verification.",
        ["No verified benefits assigned."],
        ["Misidentification may cause harmful use."],
        "No recommended use until identified.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Unknown.",
        "Obtain botanical identification."
    ),

    "Chilly": plant_info(
        "Chilli", "Capsicum species",
        "A plant producing spicy fruits used as spices and vegetables.",
        ["Fruits add flavour to food.",
         "Some varieties provide vitamin C."],
        ["Can irritate skin, eyes and digestive tract."],
        "Use confirmed edible fruits in normal food quantities.",
        ["Cooking", "Spices", "Food processing"],
        "Verified fruits are commonly eaten; do not assume leaves are edible.",
        ["Capsaicinoids including capsaicin"],
        "Excessive intake can cause burning and stomach discomfort.",
        "Wash hands after handling and avoid touching eyes."
    ),

    "Citron lime (herelikai)": plant_info(
        "Citron / Regional Citrus", "Citrus species; unconfirmed",
        "A citrus-related regional dataset label.",
        ["Some citrus fruits are used in cooking and drinks."],
        ["Citrus oils may irritate sensitive skin."],
        "Use only confirmed edible fruit.",
        ["Cooking", "Drinks", "Flavouring"],
        "Only confirmed edible citrus fruit should be eaten.",
        ["Citrus essential oils and furocoumarins"],
        "Effects depend on the species and plant part.",
        "Verify the exact citrus species."
    ),

    "Coffee": plant_info(
        "Coffee", "Coffea species",
        "An evergreen shrub producing coffee berries.",
        ["Processed beans are used to prepare beverages."],
        ["Excess caffeine may cause insomnia and palpitations."],
        "Consume commercially processed coffee in moderation.",
        ["Beverages", "Food processing"],
        "Processed coffee beans are commonly consumed.",
        ["Caffeine"],
        "Excessive caffeine may cause restlessness and rapid heartbeat.",
        "Avoid excessive caffeine and unidentified berries."
    ),

    "Common rue(naagdalli)": plant_info(
        "Common Rue", "Possibly Ruta graveolens",
        "An aromatic herb with divided leaves.",
        ["Historically used as an ornamental plant."],
        ["Can cause photosensitivity and poisoning."],
        "Do not ingest or use as a homemade remedy.",
        ["Ornamental gardening"],
        "Unsafe for self-medication. Avoid ingestion.",
        ["Furanocoumarins and other active compounds"],
        "May cause vomiting, abdominal pain and serious reactions.",
        "Avoid ingestion, especially during pregnancy."
    ),

    "Coriender": plant_info(
        "Coriander", "Coriandrum sativum",
        "An aromatic herb used in cooking.",
        ["Adds flavour to food.",
         "Provides aromatic plant compounds."],
        ["Can cause allergic reactions in sensitive individuals."],
        "Wash leaves and use in normal food quantities.",
        ["Cooking", "Spices", "Food processing"],
        "Commonly eaten as a culinary herb.",
        ["No specific toxin assigned to ordinary culinary use."],
        "Allergic individuals may develop reactions.",
        "Avoid if allergic and wash before use."
    ),

    "Curry": plant_info(
        "Curry Leaf", "Bergera koenigii",
        "A small tree with aromatic compound leaves.",
        ["Used to flavour food.",
         "Contains aromatic plant compounds."],
        ["Allergic reactions are possible."],
        "Wash and use correctly identified leaves in cooking.",
        ["Cooking", "Seasoning", "Food preparation"],
        "Commonly eaten as a culinary herb.",
        ["No specific toxin assigned to normal culinary use."],
        "Individual allergies are possible.",
        "Do not confuse with unrelated curry-scented plants."
    ),

    "Doddpathre": plant_info(
        "Doddapatre", "Possibly Coleus amboinicus",
        "An aromatic plant with thick, fleshy leaves.",
        ["Used in some regional cooking and traditional practices."],
        ["Regional names may refer to different species."],
        "Confirm species before using leaves in food or remedies.",
        ["Regional cooking", "Traditional preparations"],
        "Do not consume based only on prediction.",
        ["Species-dependent essential oils"],
        "Effects depend on actual species and preparation.",
        "Verify botanical identity."
    ),

    "Drumstick": plant_info(
        "Drumstick / Moringa", "Moringa oleifera",
        "A tree with compound leaves and long seed pods.",
        ["Leaves and pods are used as vegetables.",
         "Leaves provide nutrients."],
        ["Concentrated extracts may cause adverse effects."],
        "Wash and properly prepare edible leaves and pods.",
        ["Food", "Agriculture", "Traditional products"],
        "Certain properly identified parts are commonly eaten.",
        ["No specific toxin assigned to ordinary cooked leaves and pods."],
        "Concentrated preparations may cause adverse effects.",
        "Avoid unsupervised medicinal extracts."
    ),

    "Ekka": plant_info(
        "Calotropis / Ekka", "Possibly Calotropis gigantea",
        "A shrub with thick leaves and milky latex.",
        ["Has traditional and fibre-related uses."],
        ["Latex is irritating and the plant can be poisonous."],
        "Do not eat or prepare homemade remedies.",
        ["Fibre", "Ornamental uses"],
        "TOXIC. Do not eat any part.",
        ["Cardiac glycosides"],
        "May cause irritation, vomiting and serious heart effects.",
        "Avoid latex contact. Keep away from children and pets."
    ),

    "Eucalyptus": plant_info(
        "Eucalyptus", "Eucalyptus species",
        "A group of aromatic trees with oil-containing leaves.",
        ["Used for timber, paper and commercial products."],
        ["Essential oil can be poisonous if swallowed."],
        "Use only labelled commercial products as directed.",
        ["Timber", "Paper", "Essential oils"],
        "Do not eat leaves or essential oil.",
        ["1,8-cineole and other volatile oils"],
        "Swallowing eucalyptus oil can cause seizures and serious poisoning.",
        "Keep essential oils away from children."
    ),

    "Ganigale": plant_info(
        "Ganigale", "Scientific identity unconfirmed",
        "Regional plant label requiring botanical verification.",
        ["No verified benefits assigned."],
        ["Unknown identity creates misidentification risk."],
        "No recommended food or medicinal use.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Unknown.",
        "Verify the species with a botanist."
    ),

    "Ganike": plant_info(
        "Ganike", "Scientific identity unconfirmed",
        "Regional plant name requiring confirmation.",
        ["No reliable species-specific benefits assigned."],
        ["Misidentification may cause harmful use."],
        "Do not consume until identified.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Unknown.",
        "Consult a botanical expert."
    ),

    "Gasagase": plant_info(
        "Gasagase", "Species unconfirmed",
        "A regional name that may refer to poppy seeds or a related plant.",
        ["Some verified poppy seeds are used in cooking."],
        ["Different species and plant parts have different safety profiles."],
        "Do not use unidentified plant material as food or medicine.",
        ["Food seasoning, subject to confirmed identity"],
        "The label does not establish edibility.",
        ["Opium alkaloids occur in some poppy species and parts."],
        "Effects depend on exact species and plant part.",
        "Confirm the species and plant part."
    ),

    "Ginger": plant_info(
        "Ginger", "Zingiber officinale",
        "A herbaceous plant with narrow leaves and an aromatic rhizome.",
        ["Rhizome is widely used as a culinary spice."],
        ["May cause heartburn or digestive discomfort."],
        "Use the correctly identified and cleaned rhizome in food.",
        ["Cooking", "Beverages", "Food processing"],
        "Rhizome is commonly eaten, not necessarily the leaves.",
        ["Gingerols and related compounds"],
        "Large amounts may cause heartburn and stomach upset.",
        "Use normal food quantities."
    ),

    "Globe Amarnath": plant_info(
        "Globe Amaranth", "Gomphrena globosa",
        "An ornamental plant with round, colourful flower heads.",
        ["Grown for ornamental and decorative purposes."],
        ["Food safety of this species is not established here."],
        "Use as an ornamental plant.",
        ["Ornamental gardening", "Decorative arrangements"],
        "Unknown. Do not consume.",
        ["Species-specific composition requires verification."],
        "Ingestion effects are not established.",
        "Do not assume related amaranths share the same edibility."
    ),

    "Guava": plant_info(
        "Guava", "Psidium guajava",
        "A tropical fruit tree with aromatic leaves and edible fruit.",
        ["Fruit provides vitamin C and dietary fibre.",
         "Leaves have traditional uses."],
        ["Some people may experience allergies or digestive discomfort."],
        "Wash and eat correctly identified ripe fruit.",
        ["Food", "Juices", "Jams", "Traditional preparations"],
        "Ripe fruit is commonly eaten.",
        ["No specific toxin assigned to normal ripe fruit."],
        "Excessive fruit intake may cause digestive discomfort.",
        "Leaf extracts are not a substitute for medical treatment."
    ),

    "Henna": plant_info(
        "Henna", "Lawsonia inermis",
        "A shrub whose leaves are processed to produce natural dye.",
        ["Leaves are used in traditional hair and skin dye products."],
        ["May cause allergic reactions.",
         "Some black henna products contain harmful added chemicals."],
        "Use verified external-use products only.",
        ["Hair dye", "Body art", "Cosmetics"],
        "Do not eat henna leaves or ingest henna products.",
        ["Lawsone; adulterated products may contain additional chemicals."],
        "Ingestion can be harmful.",
        "External use only. Avoid products of uncertain composition."
    ),

    "Hibiscus": plant_info(
        "Hibiscus", "Hibiscus species; exact species unconfirmed",
        "A group of flowering plants with colourful blooms.",
        ["Some species and parts are used in food and beverages.",
         "Many species are ornamental."],
        ["Safety differs between species and plant parts."],
        "Use only food-grade material from an identified edible species.",
        ["Ornamental gardening", "Beverages", "Food products"],
        "Species- and part-dependent.",
        ["Species-specific compounds require verification."],
        "Effects depend on actual species and preparation.",
        "Verify the exact species before ingestion."
    ),

    "Honge": plant_info(
        "Indian Beech / Honge", "Millettia pinnata",
        "A tree with compound leaves and oil-bearing seeds.",
        ["Used for shade, soil improvement and industrial oil."],
        ["Seeds and unprocessed oil are not suitable for casual consumption."],
        "Do not ingest unprocessed plant products.",
        ["Shade", "Agroforestry", "Industrial oil"],
        "Do not eat seeds or leaves based on prediction.",
        ["Karanjin and pongamol"],
        "Ingestion of unprocessed parts may cause adverse effects.",
        "Keep seeds and oils away from children and pets."
    ),

    "Insulin": plant_info(
        "Insulin Plant", "Possibly Chamaecostus cuspidatus",
        "A plant marketed in some regions as an insulin plant.",
        ["Has a history of traditional use."],
        ["Diabetes-treatment claims are not established by identification.",
         "May interact with diabetes medicines."],
        "Never replace prescribed diabetes medication with this plant.",
        ["Ornamental gardening", "Traditional herbal use"],
        "Do not consume based on prediction.",
        ["Plant constituents require verification."],
        "Possible interactions with blood glucose treatment.",
        "Consult a healthcare professional."
    ),

    "Jackfruit": plant_info(
        "Jackfruit", "Artocarpus heterophyllus",
        "A tropical tree producing large edible fruits.",
        ["Fruit provides energy and dietary fibre.",
         "Seeds are used as food after cooking."],
        ["May cause digestive discomfort or allergies."],
        "Eat properly identified ripe fruit; cook seeds thoroughly.",
        ["Food", "Cooking", "Food processing"],
        "Ripe fruit and properly cooked seeds are commonly eaten.",
        ["No specific toxin assigned to normal ripe fruit."],
        "Improperly prepared parts may cause digestive problems.",
        "Do not eat seeds raw. Check for allergies."
    ),

    "Jasmine": plant_info(
        "Jasmine", "Jasminum species",
        "A group of flowering shrubs and climbers known for fragrance.",
        ["Flowers are used in fragrance and ornamental products."],
        ["Edibility and safety differ between species."],
        "Do not consume unidentified plant parts.",
        ["Perfumes", "Ornamental gardening", "Floral products"],
        "Do not assume leaves or flowers are edible.",
        ["Species-specific composition requires verification."],
        "Cannot be predicted without exact identification.",
        "Confirm the botanical species."
    ),

    "Kambajala": plant_info(
        "Kambajala", "Scientific identity unconfirmed",
        "Regional name requiring botanical identification.",
        ["No species-specific benefits verified."],
        ["Potential misidentification risk."],
        "No recommended use until confirmed.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Unknown.",
        "Obtain the scientific name from an expert."
    ),

    "Kasambruga": plant_info(
        "Kasambruga", "Scientific identity unconfirmed",
        "Regional dataset name requiring confirmation.",
        ["No verified benefits assigned."],
        ["Misidentification may lead to harmful use."],
        "Do not consume until identified.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Unknown.",
        "Verify the scientific name."
    ),

    "Kohlrabi": plant_info(
        "Kohlrabi", "Brassica oleracea var. gongylodes",
        "A vegetable with a swollen edible stem and leafy tops.",
        ["Provides fibre and vitamin C.",
         "Used as a vegetable."],
        ["May cause gas or bloating."],
        "Wash and cook or prepare properly identified produce.",
        ["Food", "Cooking", "Agriculture"],
        "Commonly eaten as a vegetable.",
        ["Glucosinolates"],
        "Normal food quantities are generally tolerated.",
        "Wash thoroughly and confirm identity."
    ),

    "Lantana": plant_info(
        "Lantana", "Lantana camara",
        "A flowering shrub commonly grown as an ornamental plant.",
        ["Used as an ornamental plant."],
        ["Can be invasive and toxic to people and animals."],
        "Ornamental use only. Never eat berries or leaves.",
        ["Ornamental gardening"],
        "TOXICITY RISK. Do not eat any part.",
        ["Triterpenoids including lantadene compounds"],
        "May cause vomiting, diarrhea and liver-related toxicity.",
        "Keep away from children and pets."
    ),

    "Lemon": plant_info(
        "Lemon", "Citrus limon",
        "A citrus tree producing sour fruits.",
        ["Fruit provides vitamin C.",
         "Used in cooking and drinks."],
        ["Acidic juice may irritate sensitive skin and teeth."],
        "Use washed, correctly identified fruit.",
        ["Cooking", "Beverages", "Food processing"],
        "Fruit is commonly eaten.",
        ["Citric acid and citrus oils"],
        "Large amounts may irritate the stomach.",
        "Avoid eye contact with juice."
    ),

    "Malabar_Nut": plant_info(
        "Malabar Nut", "Justicia adhatoda",
        "A shrub traditionally used in respiratory preparations.",
        ["Has a history of traditional herbal use."],
        ["Medicinal preparations may have contraindications."],
        "Do not self-medicate with leaf extracts.",
        ["Traditional herbal preparations"],
        "Do not consume based on prediction.",
        ["Vasicine and related alkaloids"],
        "Effects depend on dose and preparation.",
        "Seek medical advice, especially during pregnancy."
    ),

    "Malabar_Spinach": plant_info(
        "Malabar Spinach", "Basella alba",
        "A climbing leafy vegetable with thick, fleshy leaves.",
        ["Provides leafy vegetables for cooking.",
         "Contains fibre and micronutrients."],
        ["May cause digestive discomfort."],
        "Wash and cook correctly identified leaves and shoots.",
        ["Food", "Cooking", "Home gardening"],
        "Leaves and shoots are commonly eaten.",
        ["No specific toxin assigned to ordinary cooked portions."],
        "Large quantities may cause digestive discomfort.",
        "Confirm species and wash thoroughly."
    ),

    "Mango": plant_info(
        "Mango", "Mangifera indica",
        "A tropical tree grown for its sweet fruit.",
        ["Fruit provides vitamins and dietary fibre."],
        ["Sap and peel may cause contact dermatitis."],
        "Eat washed, ripe fruit. Do not eat leaves based on prediction.",
        ["Food", "Juices", "Pickles"],
        "Ripe fruit is commonly eaten.",
        ["Urushiol-related compounds in sap and peel"],
        "Unsuitable plant parts may cause irritation.",
        "Avoid sap contact if sensitive."
    ),

    "Marigold": plant_info(
        "Marigold", "Tagetes species",
        "A group of flowering plants grown in gardens.",
        ["Used ornamentally and in horticulture."],
        ["May cause skin irritation or allergic reactions."],
        "Do not eat unless the exact species is confirmed food-safe.",
        ["Ornamental gardening", "Horticulture"],
        "Do not assume this species is edible.",
        ["Essential oils and thiophenes vary by species."],
        "Effects depend on species and quantity.",
        "Never eat pesticide-treated garden flowers."
    ),

    "Mint": plant_info(
        "Mint", "Mentha species",
        "An aromatic herb used in cooking and beverages.",
        ["Some mint leaves are used in food.",
         "Adds aroma and flavour."],
        ["Can worsen reflux in some people.",
         "Concentrated oils can be harmful."],
        "Use positively identified culinary mint leaves.",
        ["Cooking", "Beverages", "Food flavouring"],
        "Some verified mint species are edible.",
        ["Menthol and other essential-oil compounds"],
        "Large amounts of concentrated oil may cause poisoning.",
        "Do not swallow concentrated essential oils."
    ),

    "Neem": plant_info(
        "Neem", "Azadirachta indica",
        "A tropical tree with compound leaves and bitter plant parts.",
        ["Used in agricultural and topical products.",
         "Has traditional uses."],
        ["Neem oil can be poisonous if swallowed."],
        "Use only labelled products as directed.",
        ["Agriculture", "Pest management", "Topical products"],
        "Do not ingest neem oil or unidentified preparations.",
        ["Azadirachtin and other limonoids"],
        "Swallowing neem oil can cause seizures and serious poisoning.",
        "Keep neem oil away from children."
    ),

    "Nelavembu": plant_info(
        "King of Bitters", "Possibly Andrographis paniculata",
        "A bitter herb used in traditional preparations.",
        ["Has a history of traditional herbal use."],
        ["May cause gastrointestinal effects and allergic reactions."],
        "Do not self-medicate with concentrated extracts.",
        ["Traditional herbal products"],
        "Do not consume based on prediction.",
        ["Andrographolides"],
        "May cause nausea, vomiting, diarrhea or allergic reactions.",
        "Check for medicine interactions and seek professional advice."
    ),

    "Nerale": plant_info(
        "Jamun", "Possibly Syzygium cumini",
        "A tree cultivated for dark purple edible fruits.",
        ["Fruit is used in food and beverages."],
        ["Excessive intake may cause digestive discomfort."],
        "Eat only correctly identified ripe fruit.",
        ["Food", "Juices"],
        "Ripe fruit is commonly eaten.",
        ["No specific toxin assigned to normal ripe fruit."],
        "Excessive intake may cause digestive discomfort.",
        "Do not use extracts to manage diabetes without medical advice."
    ),

    "Nooni": plant_info(
        "Noni", "Possibly Morinda citrifolia",
        "A tropical plant whose fruit is used in some products.",
        ["Fruit is used in some food and commercial preparations."],
        ["Some concentrated products have been associated with liver concerns."],
        "Do not use concentrated products as medicine without advice.",
        ["Food products", "Commercial preparations"],
        "Do not consume based on prediction.",
        ["Composition varies by product."],
        "Adverse effects and liver concerns have been reported with some products.",
        "Consult a healthcare professional."
    ),

    "Onion": plant_info(
        "Onion", "Allium cepa",
        "A cultivated vegetable with a bulb and green leaves.",
        ["Bulbs are widely used in cooking."],
        ["May cause digestive discomfort."],
        "Use washed, correctly identified bulbs in food.",
        ["Cooking", "Food processing", "Agriculture"],
        "Bulbs are commonly eaten.",
        ["Organosulfur compounds"],
        "Excess intake may cause bloating.",
        "Do not assume an unknown bulb is onion."
    ),

    "Padri": plant_info(
        "Padri", "Scientific identity unconfirmed",
        "Regional name requiring botanical verification.",
        ["No species-specific benefits verified."],
        ["Unknown identity makes use unsafe."],
        "No recommended use until identified.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Unknown.",
        "Confirm the scientific name."
    ),

    "Palak(Spinach)": plant_info(
        "Spinach", "Spinacia oleracea",
        "A leafy vegetable cultivated for its edible leaves.",
        ["Provides folate and vitamin K.",
         "Used in cooked dishes."],
        ["Contains oxalates, relevant to some people with kidney stone risk."],
        "Wash thoroughly and prepare as a vegetable.",
        ["Cooking", "Food processing", "Agriculture"],
        "Commonly eaten as a leafy vegetable.",
        ["Oxalates"],
        "Excessive intake may not suit everyone.",
        "Follow individualized dietary advice when medically necessary."
    ),

    "Papaya": plant_info(
        "Papaya", "Carica papaya",
        "A tropical plant producing large edible fruits.",
        ["Ripe fruit provides vitamin C and dietary fibre."],
        ["Latex can irritate skin.",
         "Unripe papaya preparations may be unsuitable during pregnancy."],
        "Eat properly identified ripe fruit.",
        ["Food", "Juices", "Food processing"],
        "Ripe fruit is commonly eaten.",
        ["Papain and latex-related compounds"],
        "Unripe latex-containing parts may cause adverse reactions.",
        "Seek medical advice before medicinal use, especially during pregnancy."
    ),

    "Parijatha": plant_info(
        "Night-flowering Jasmine", "Possibly Nyctanthes arbor-tristis",
        "A small tree or shrub with fragrant flowers.",
        ["Flowers are used ornamentally and traditionally."],
        ["Medicinal safety and effectiveness are not established for home use."],
        "Use as an ornamental plant; avoid homemade remedies.",
        ["Ornamental gardening", "Traditional preparations"],
        "Do not consume based on prediction.",
        ["Species-specific composition requires verification."],
        "Effects depend on plant part and preparation.",
        "Verify species before medicinal use."
    ),

    "Pea": plant_info(
        "Pea", "Pisum sativum",
        "A legume grown for edible pods and seeds.",
        ["Provides plant protein and dietary fibre."],
        ["May cause gas or digestive discomfort."],
        "Wash and properly prepare identified peas and pods.",
        ["Food", "Agriculture", "Food processing"],
        "Commonly eaten.",
        ["No specific toxin assigned to normal edible peas."],
        "Excess intake may cause bloating.",
        "Confirm identity before eating unfamiliar pods."
    ),

    "Pomoegranate": plant_info(
        "Pomegranate", "Punica granatum",
        "A shrub or small tree producing fruit with juicy seed coverings.",
        ["Fruit provides fibre and plant pigments."],
        ["Excessive intake may cause digestive discomfort."],
        "Eat correctly identified ripe fruit.",
        ["Food", "Juices", "Food processing"],
        "Ripe fruit arils are commonly eaten.",
        ["No specific toxin assigned to ordinary ripe arils."],
        "Excessive intake may cause digestive discomfort.",
        "Do not assume leaves and bark have the same food uses."
    ),

    "Pumpkin": plant_info(
        "Pumpkin", "Cucurbita species",
        "A creeping plant producing large fruits.",
        ["Some varieties provide edible fruit, seeds and flowers."],
        ["Some cucurbits can contain toxic bitter compounds."],
        "Eat only verified food varieties. Discard unusually bitter fruit.",
        ["Food", "Cooking", "Agriculture"],
        "Variety- and plant-part-dependent.",
        ["Cucurbitacins in some bitter cucurbits"],
        "Toxic cucurbitacins can cause severe vomiting and diarrhea.",
        "Never eat unusually bitter pumpkin or gourd."
    ),

    "Raddish": plant_info(
        "Radish", "Raphanus sativus",
        "A root vegetable with edible roots and some edible leafy varieties.",
        ["Provides fibre and vitamin C."],
        ["May cause gas or digestive discomfort."],
        "Wash and prepare correctly identified roots and edible leaves.",
        ["Cooking", "Food", "Agriculture"],
        "Roots are commonly eaten; some leaves are also used as food.",
        ["Glucosinolates"],
        "Large quantities may cause digestive discomfort.",
        "Confirm the variety and wash thoroughly."
    ),

    "Rose": plant_info(
        "Rose", "Rosa species",
        "A group of flowering shrubs grown for ornamental purposes.",
        ["Flowers are used in perfumes and some food products."],
        ["Thorns can injure skin.",
         "Ornamental flowers may contain pesticide residues."],
        "Only use correctly identified food-grade petals.",
        ["Perfumes", "Cosmetics", "Ornamental gardening", "Food products"],
        "Some petals are used in food; leaves should not be assumed edible.",
        ["No single harmful compound applies to all roses."],
        "Effects depend on species and pesticide exposure.",
        "Never eat flowers from unknown or chemically treated gardens."
    ),

    "Sampige": plant_info(
        "Champak", "Possibly Magnolia champaca",
        "A tree known for fragrant flowers.",
        ["Flowers are used in perfumes and cultural practices."],
        ["Food safety of leaves and flowers is not established."],
        "Use for ornamental and fragrance purposes.",
        ["Perfumes", "Ornamental gardening", "Cultural uses"],
        "Do not assume any part is edible.",
        ["Species-specific composition requires verification."],
        "Ingestion effects are not established.",
        "Confirm the exact species before consumption."
    ),

    "Sapota": plant_info(
        "Sapota / Chikoo", "Manilkara zapota",
        "A tropical fruit tree producing sweet brown fruits.",
        ["Ripe fruit provides energy and dietary fibre."],
        ["Unripe fruit may cause digestive discomfort."],
        "Eat properly identified, ripe fruit.",
        ["Food", "Desserts", "Beverages"],
        "Ripe fruit is commonly eaten.",
        ["No specific toxin assigned to normal ripe fruit."],
        "Unripe fruit or excess intake may cause digestive discomfort.",
        "Do not consume seeds or unidentified plant parts."
    ),

    "Seethaashoka": plant_info(
        "Ashoka-related Tree", "Species unconfirmed",
        "A regional name that may refer to different trees.",
        ["Some similarly named trees are used ornamentally."],
        ["Confusing different species can lead to incorrect use."],
        "No medicinal or food use until identified.",
        ["Ornamental gardening"],
        "Unknown. Do not consume.",
        ["Unknown until species is confirmed."],
        "Cannot be predicted reliably.",
        "Confirm the scientific name."
    ),

    "Seethapala": plant_info(
        "Custard Apple", "Possibly Annona squamosa",
        "A fruit tree producing sweet segmented fruits.",
        ["Ripe fruit pulp is used as food."],
        ["Seeds should not be eaten."],
        "Eat only ripe pulp and discard seeds.",
        ["Food", "Desserts", "Beverages"],
        "Ripe pulp is commonly eaten; seeds must not be eaten.",
        ["Annonaceous acetogenins in seeds and other parts."],
        "Seeds may be toxic.",
        "Never chew or swallow seeds."
    ),

    "Spinach1": plant_info(
        "Spinach (dataset variant)", "Scientific identity unconfirmed",
        "Dataset label does not establish the exact spinach species.",
        ["Some spinach species are used as leafy vegetables."],
        ["Different leafy plants have different safety profiles."],
        "Do not consume until the species is confirmed.",
        ["Potential food use, subject to identification"],
        "Unknown until species is confirmed.",
        ["Unknown"],
        "Cannot be reliably predicted.",
        "Verify the scientific name before use."
    ),

    "Tamarind": plant_info(
        "Tamarind", "Tamarindus indica",
        "A tropical tree producing pods with sour-sweet pulp.",
        ["Pulp is used in cooking and beverages."],
        ["Excessive intake may cause digestive discomfort."],
        "Use properly identified ripe pulp in normal quantities.",
        ["Cooking", "Beverages", "Food processing"],
        "Ripe pulp is commonly eaten.",
        ["Organic acids in fruit pulp"],
        "Excessive intake may cause stomach discomfort.",
        "Do not assume leaves, bark and seeds have the same food uses."
    ),

    "Taro": plant_info(
        "Taro", "Colocasia esculenta",
        "A tropical plant grown for its starchy underground corms.",
        ["Properly prepared corms and some leaves are used as food."],
        ["Raw parts can cause severe mouth and throat irritation."],
        "Only eat correctly identified parts after thorough appropriate cooking.",
        ["Food", "Agriculture"],
        "Edible only with correct identification and preparation.",
        ["Calcium oxalate raphides"],
        "Raw taro may cause intense burning and swelling of the mouth and throat.",
        "Never eat raw taro."
    ),

    "Tecoma": plant_info(
        "Tecoma", "Tecoma species",
        "A group of flowering shrubs and small trees.",
        ["Primarily used for ornamental gardening."],
        ["Edibility is not established for this identification."],
        "Ornamental use only.",
        ["Ornamental gardening"],
        "Unknown. Do not consume.",
        ["Species-specific compounds require verification."],
        "Ingestion effects are not established.",
        "Do not use as food or medicine based on prediction."
    ),

    "Thumbe": plant_info(
        "Thumbe", "Possibly Leucas species",
        "A regional name used for certain flowering herbs.",
        ["Some species have traditional uses."],
        ["Regional names may refer to more than one species."],
        "Do not prepare herbal remedies without botanical confirmation.",
        ["Traditional preparations"],
        "Unknown. Do not consume.",
        ["Species-specific composition requires verification."],
        "Effects depend on actual species and preparation.",
        "Confirm the scientific name."
    ),

    "Tomato": plant_info(
        "Tomato", "Solanum lycopersicum",
        "A cultivated plant producing edible fruits.",
        ["Ripe fruit provides vitamin C and lycopene."],
        ["Green plant parts are not food and may cause illness."],
        "Eat ripe, washed fruits. Avoid leaves and stems.",
        ["Cooking", "Food processing", "Agriculture"],
        "Ripe fruits are commonly eaten; leaves and stems should not be eaten.",
        ["Tomatine and related glycoalkaloids"],
        "Eating leaves or stems may cause nausea and digestive symptoms.",
        "Use ripe fruit only."
    ),

    "Tulsi": plant_info(
        "Holy Basil", "Ocimum tenuiflorum",
        "An aromatic herb used in cooking and traditional practices.",
        ["Used in foods and beverages.",
         "Has a history of traditional use."],
        ["Concentrated preparations may cause interactions."],
        "Use correctly identified leaves in normal culinary quantities.",
        ["Cooking", "Beverages", "Traditional preparations"],
        "Commonly used as a culinary herb.",
        ["Eugenol and other essential-oil compounds"],
        "Excessive concentrated preparations may cause unwanted effects.",
        "Do not replace prescribed treatment with Tulsi."
    ),

    "Turmeric": plant_info(
        "Turmeric", "Curcuma longa",
        "A herbaceous plant with an underground yellow-orange rhizome.",
        ["Rhizome is widely used as a culinary spice."],
        ["High-dose supplements may cause digestive effects or interactions."],
        "Use correctly identified rhizome in food.",
        ["Cooking", "Food colouring", "Food processing"],
        "Rhizome is commonly eaten as a spice.",
        ["Curcuminoids"],
        "Excessive supplement use may cause digestive upset.",
        "Food use differs from concentrated supplements."
    ),

    "ashoka": plant_info(
        "Ashoka", "Possibly Saraca asoca",
        "A tree with compound leaves and clusters of flowers.",
        ["Grown for ornamental and cultural purposes."],
        ["Misidentification can lead to incorrect medicinal use."],
        "Do not use medicinally until the species is verified.",
        ["Ornamental gardening", "Traditional preparations"],
        "Unknown. Do not consume.",
        ["Species-specific compounds require verification."],
        "Effects cannot be predicted without identification.",
        "Do not confuse true Ashoka with unrelated trees."
    ),

    "camphor": plant_info(
        "Camphor", "Possibly Cinnamomum camphora",
        "An aromatic compound used in some topical and fragrance products.",
        ["Used in some topical and fragrance products."],
        ["Can cause severe poisoning if swallowed."],
        "Use only appropriately labelled external products.",
        ["Fragrance", "Topical products", "Cultural uses"],
        "TOXIC. Never eat camphor.",
        ["Camphor"],
        "Swallowing camphor may cause vomiting, seizures and life-threatening poisoning.",
        "Keep away from children and pets."
    ),

    "kamakasturi": plant_info(
        "Kamakasturi", "Species unconfirmed",
        "Regional name associated with aromatic plants.",
        ["Some related plants are used in food or traditional preparations."],
        ["Similar regional names may refer to different species."],
        "Do not consume until the exact species is verified.",
        ["Potential culinary or traditional use, subject to confirmation"],
        "Unknown until species is confirmed.",
        ["Unknown"],
        "Effects cannot be predicted reliably.",
        "Confirm the scientific name."
    ),

    "kepala": plant_info(
        "Kepala", "Scientific identity unconfirmed",
        "Regional dataset name requiring botanical verification.",
        ["No verified species-specific benefits assigned."],
        ["Incorrect identification may result in unsafe use."],
        "No recommended use until species is confirmed.",
        ["Not established"],
        "Unknown. Do not consume.",
        ["Unknown"],
        "Unknown.",
        "Obtain a confirmed scientific name from a botanical expert."
    ),


}
DEFAULT_PLANT_INFO = {
    "common_name": "Unknown plant",
    "scientific_name": "Not verified",
    "description": "Plant identity requires verification.",
    "benefits": ["Information not verified."],
    "disadvantages": ["Incorrect identification may be unsafe."],
    "proper_usage": "Do not consume until identified.",
    "applications": ["Not established"],
    "uses": "Information not available.",
    "edibility": "UNKNOWN. Do not consume based on AI prediction.",
    "harmful_compounds": ["Unknown"],
    "if_consumed": "Effects cannot be determined.",
    "precautions": "Consult a botanical expert.",
    "safety": "Do not consume until the species is verified."
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
