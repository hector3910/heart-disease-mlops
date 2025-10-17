from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_predict_endpoint():
    # ejemplo de entrada (debe tener los mismos 11 features)
    data = {
        "features": [40, "M", "ATA", 140, 289, 0, "Normal", 172, "N", 0.0, "Up"]
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "heart_disease_probability" in result
    assert "prediction" in result
