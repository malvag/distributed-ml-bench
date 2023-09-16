import requests

input_path = "inference-input.json"

response = requests.post(url="http://localhost:8080/v1/models/flower-sample:predict", data=json.dumps(data), headers={'Host': 'flower-sample.kubeflow.example.com'})

print(response.text)
