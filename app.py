from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import spacy

SPACY_MODELS = {
    "fr": "fr_core_news_md", # French: medium-sized model
    "en": "en_core_web_md"   # English: medium-sized model
}

TRANSLATORS = {}

app = Flask(__name__)
CORS(app)

# TEST Route to verify functioning of API / Port-forwarding
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Flask on Ubuntu VM!"})

@app.route('/translate-analyse', methods=['POST'])
def translate_analyse():
    data = request.get_json()
    text = data.get("text", "").strip() # Might need to parse into body and title
    source = data.get("source_lang")
    target = data.get("target_lang")

    if not text or not source or not target:
        return jsonify({"error": "Missing text or language codes"}), 400

    try:
        nlp = get_spacy(source)
        translator = get_translator(source, target)
    except Exception as e:
        return jsonify({"error":f"Language not supported or failed to load model: {e}"}), 500

    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        s_text = sent.text.strip()
        translation = translator(s_text, max_length=512)[0]['translation_text']
        tokens = [{"text": tok.text, "pos": tok.pos_} for tok in sent]
        sentences.append({
            "original": s_text,
            "translation": translation,
            "tokens": tokens
        })

    return jsonify({
        "sentences": sentences,
        "source_lang": source,
        "target_lang": target
    })


def get_spacy(lang_code):
    return spacy.load(SPACY_MODELS[lang_code])

def get_translator(source, target):
    key = f"{source}_to_{target}"
    if key not in TRANSLATORS:
        model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
        TRANSLATORS[key] = pipeline("translation", model=model_name)
    return TRANSLATORS[key]

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5000, debug=True)