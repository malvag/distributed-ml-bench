import requests

response = requests.post(url="http://localhost:8080/v1/models/tf-mnist:predict", json=mnist-input.json, headers={'Host': 'tf-mnist.kubeflow.example.com'})
print(response.text)
