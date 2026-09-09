import requests
import pandas as pd

BASE_URL = "https://faostatservices.fao.org/api/v1"

# ------------------------------------------------------------
# Your FAOSTAT credentials
# ------------------------------------------------------------

username = "manasa.savnur@gmail.com"
password = "Faostat@123"

# ------------------------------------------------------------
# Login and get AccessToken
# ------------------------------------------------------------

login_response = requests.post(
    f"{BASE_URL}/auth/login",
    data={
        "username": username,
        "password": password
    },
    timeout=30
)

login_response.raise_for_status()

access_token = login_response.json()["AuthenticationResult"]["AccessToken"]

headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json"
}

print("Login successful.")

# ------------------------------------------------------------
# Get Trade Indices data for 2023
# ------------------------------------------------------------

response = requests.get(
    f"{BASE_URL}/en/data/TI",
    headers=headers,
    params={
        "year": "2023",
        "output_type": "objects",
        "limit": 5
    },
    timeout=60
)

response.raise_for_status()

# ------------------------------------------------------------
# Convert response to DataFrame
# ------------------------------------------------------------

data = response.json()["data"]

df = pd.DataFrame(data)

# Convert numeric fields
numeric_columns = [
    "Value",
    "Year",
    "Area Code",
    "Element Code",
    "Item Code"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print("\nDataFrame:")
print(df)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())