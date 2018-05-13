import requests
from datetime import datetime
import const
import settings

config = settings.config[const.DEFAULT]

result_path = config[const.SUBMIT_DIR] + "result___" + datetime.utcnow().strftime("%Y_%m_%d") + ".csv"
files = {'files': open(result_path, 'rb')}

data = {
    "user_id": "pouyaesm",
    # user_id is your username which can be found on the top-right corner
    # on our website when you logged in.
    "team_token": "",  # your team_token.
    "description": 'Hybrid model with pre-train',  # no more than 40 chars.
    "filename": "submit_" + datetime.utcnow().date().strftime("%Y_%m_%d_%H_%M_%S"),  # your filename
}

url = 'https://biendata.com/competition/kdd_2018_submit/'

response = requests.post(url, files=files, data=data)

print(response.text)
