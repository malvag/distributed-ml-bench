# distributed-ml-system

## Setup

[1] TensorFlow
- `pip install tensorflow`

[2] docker

[3] kubectl
- `brew install kubectl`

[4] k3d.io
- `wget -q -O - https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=v5.0.0 bash`
- `k3d cluster create mycluster`
- `kubectl get nodes`

[5] kubectx and kubens
- `brew install kubectx`

For example if you want to create a kubernetes pod, then create a hello-world.yaml and then do `kubectl create -f hello-world.yaml` 

```
apiVersion: v1
kind: Pod
metadata:
  name: whalesay
spec:
  containers:
  - name: whalesay
    image: docker/whalesay:latest
    command: [cowsay]
    args: ["hello world"]
```

then `kubectl get pods`

to see what is being printed out in the contianer, you can do  `kubectl logs whalesay`

If you want to get the details of a single pod with the raw yaml, then do `kubectl get pod whalesay -o yaml`
You can get the JSON or any other format as well


## Introduction

We are building an image classification end-to-end machine learning system.

### System Architecture

<img width="1143" alt="Screenshot 2023-06-30 at 12 50 13 PM" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/18bb1322-1970-4ef4-a3a6-f7d345623ee0">

## Data Ingestion

We will use the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:

![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/9356978d-f3ab-4404-b35b-d5e50b3c82cb)

Here, 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network learned to classify images.


## Model Training

## Model Serving

## End-to-end Workflow
