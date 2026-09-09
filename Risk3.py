import requests
import faostat
import pandas as pd

BASE_URL = "https://faostatservices.fao.org/api/v1"

# ------------------------------------------------------------
# Your FAOSTAT credentials
# ------------------------------------------------------------

username = "manasa.savnur@gmail.com"
password = "Faostat@123"

# ------------------------------------------------------------
# Login
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

auth = login_response.json()["AuthenticationResult"]

access_token = auth["AccessToken"]

print("Login successful.")
print("Token expires in:", auth["ExpiresIn"], "seconds")

# ------------------------------------------------------------
# Give the token to the FAOSTAT library
# ------------------------------------------------------------

faostat.set_requests_args(
    token=access_token,
    lang="en",
    timeout=120
)

# ------------------------------------------------------------
# Get ALL datasets
# ------------------------------------------------------------

datasets = faostat.list_datasets()
datasets_df = pd.DataFrame(datasets)

output_file = "faostat_datasets.csv"
datasets_df.to_csv(output_file, index=False)
print(f"Saved dataset list to {output_file}")


