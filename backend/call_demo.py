import requests, json
r = requests.get('http://127.0.0.1:8000/demo/mock?buyer_scenario=clean')
print(json.dumps(r.json(), indent=2))
