import json
import requests

groupname = "One_Direction"
filename = "results_clip_vitL14.json"
url = "http://tatooine.disi.unitn.it:3001/retrieval/"

with open(filename, "r") as f:
    results = json.load(f)

res = {
    "groupname": groupname,
    "images": results
}
res_json = json.dumps(res)

response = requests.post(url, res_json)

try:
    result = json.loads(response.text)
    print(f"Accuracy is {result['accuracy']}")
except json.JSONDecodeError:
    print(f"ERROR: {response.text}")
