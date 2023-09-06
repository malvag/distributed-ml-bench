# Distributed ML Systems

Distributing machine learning systems allows developers to handle extremely large datasets across multiple clusters, take advantage of automation tools, and benefit from hardware accelerations. This repo includes code and references to implement a scalable and reliable machine learning system.

We will automate machine learning tasks with Kubernetes, Argo Workflows, Kubeflow, and TensorFlow.

Our goal is to construct machine learning pipelines with data ingestion, distributed training, model serving, managing, and monitoring these workloads.
 
## Setup

I'm using a mac and brew to install the tools. We are going to install Tensorflow, Docker, kubectl, and k3d which is a lightweight wrapper for k3s which is lightweight Kubernetes.

[1] We will be using [TensorFlow](https://www.tensorflow.org) for data processing, model building and evaluation
- `pip install tensorflow`

[2] [Docker](https://docker-curriculum.com/#setting-up-your-computer) to create single- or multi-node [k3s](https://k3s.io) clusters

[3] kubectl is a CLI for Kubernetes
- `brew install kubectl`

[4] We will use Kubernetes as our core distributed infrastructure. In fact we will use [k3d](https://k3d.io/v5.5.2/) which is a lightweight wrapper to run k3s (Rancher Lab’s minimal Kubernetes distribution) in docker

To install k3d:
- `wget -q -O - https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=v5.0.0 bash`
- `k3d cluster create dist-ml`
- `kubectl get nodes`

[5] [kubectx](https://github.com/ahmetb/kubectx/) and kubens to easily switch contexts and namespaces
- `brew install kubectx`

[6] We will use [Kubeflow](https://www.kubeflow.org) to submit jobs to the Kubernetes cluster

![Kubeflow UI](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/aa731d8d-93cf-4089-a7a4-4a9b0f47e4eb "https://www.kubeflow.org/docs/started/architecture/")

[7] We wil also use [Argo workflows](https://argoproj.github.io/workflows) to construct and submit end to end machine learning workflows

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

to see what is being printed out in the container, you can do  `kubectl logs whalesay`

If you want to get the details of a single pod with the raw yaml, then do `kubectl get pod whalesay -o yaml`
You can get the JSON or any other format as well.


## Introduction

We are building an image classification end-to-end machine learning system. 

### System Architecture

<img width="1143" alt="Screenshot 2023-06-30 at 12 50 13 PM" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/18bb1322-1970-4ef4-a3a6-f7d345623ee0">

## Data Ingestion

We will use the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:

![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/9356978d-f3ab-4404-b35b-d5e50b3c82cb)

Here, 60,000 images are used to train the network and 10,000 images are used to evaluate how accurately the network learned to classify images.

#### Single-node Data Pipeline

The `tf.data` API enables you to build complex input pipelines from simple, reusable pieces. It's very efficient. It makes it possible to handle large amounts of data, read from different data formats, and perform complex transformations.

Load the fashion-mnist dataset into a `tf.data.Dataset` object and do some preprocessing. We normalize the image pixel values from the [0, 255] range to the [0, 1] range. We are keeping an in-memory cache to improve performance. We also shuffle the training data.

```
import tensorflow_datasets as tfds
import tensorflow as tf

def make_datasets():
 BUFFER_SIZE=10000
 def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label
 datasets, info = tfds.load(name='mnist, with_info=True, as_supervised=True)
 mnist_train = datasets['train']
 return mnist_train.map(scale).cache().shuffle(BUFFER_SIZE)
```

We have used the tensorflow_datasets module which contains a collection of datasets ready to use. This gives us a shuffled dataset where each element consists of images and labels.

#### Distributed Data Pipeline

We can consume our dataset in a distributed fashion as well and to do that we can use the same function we created before.

## Model Training

## Model Serving

## End-to-end Workflow
