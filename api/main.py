import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from groq import Groq
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee.")

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age en annees")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Temperature en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Presence de toux")
    fatigue: bool = Field(..., description="Presence de fatigue")
    maux_tete: bool = Field(..., description="Presence de maux de tete")
    region: str = Field(..., description="Region du Senegal")

class DiagnosticOutput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    confiance: str = Field(..., description="Niveau de confiance")
    message: str = Field(..., description="Recommandation")

app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modele charge : {type(model)._name_}")
print(f"classes : {list(model.classes_)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0, confiance="aucune", message=f"Sexe invalide : {patient.sexe}")

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        region_enc = 0

    features = np.array([[patient.age, sexe_enc, patient.temperature, patient.tension_sys, int(patient.toux), int(patient.fatigue), int(patient.maux_tete), region_enc]])

    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    messages = {
        "palu": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typh": "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuez a surveiller."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )

@app.get("/model-info")
def model_info():
    return {
        "type": type(model)._name_,
        "n_arbres": model.n_estimators,
        "classes": list(model.classes_),
        "n_features": model.n_features_in_
    }

class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    temperature: float
    region: str

class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Explique le diagnostic en francais simple.
Sois rassurant mais recommande une consultation.
Maximum 3 phrases. Ne fais JAMAIS de diagnostic toi-meme."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    if not groq_client:
        return ExplainOutput(explication="Service indisponible.", modele_llm="aucun")
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature}C\n"
        f"Diagnostic : {data.diagnostic} (probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur : {str(e)}"
    return ExplainOutput(explication=explication)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")
