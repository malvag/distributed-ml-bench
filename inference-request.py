import requests
import json

input_path = "mnist-input.json"

with open(input_path) as json_file:
    data = json.load(json_file)

r = requests.post(
    url="http://localhost:8080/v1/models/tf-mnist:predict",
    data=json.dumps(data),
    headers={"Host": "tf-mnist.kubeflow.example.com"},
)
print(r.text)
