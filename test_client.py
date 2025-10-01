import requests

url = "http://127.0.0.1:8000/predict"

patient_data = {
    "Pregnancies": 2,
    "Glucose": 150,
    "BloodPressure": 85,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 30.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 42
}

response = requests.post(url, json=patient_data)

print("Status code:", response.status_code)
print("Response:", response.json())
