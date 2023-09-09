# Distributed ML Systems

Recently, I was involved in a classification-based ML project where we developed a distributed scalable ML service. So here I will be building a scalable ML app with fashion-mnist. Let's get into it.

Distributing machine learning systems allows developers to handle extremely large datasets across multiple clusters, take advantage of automation tools, and benefit from hardware accelerations. This repo includes code and references to implement a scalable and reliable machine learning system.

We will automate machine learning tasks with Kubernetes, Argo Workflows, Kubeflow, and TensorFlow.

Our goal is to construct machine learning pipelines with data ingestion, distributed training, model serving, managing, and monitoring these workloads.
 
## Setup

I'm using a mac and brew to install the tools. We are going to install Tensorflow, Docker, kubectl, and k3d which is a lightweight wrapper for k3s which is lightweight Kubernetes.

[1] We will be using [TensorFlow](https://www.tensorflow.org) for data processing, model building and evaluation
```bash
pip install tensorflow
```

[2] [Docker](https://docker-curriculum.com/#setting-up-your-computer) to create single- or multi-node [k3s](https://k3s.io) clusters

[3] kubectl is a CLI for Kubernetes
```bash
brew install kubectl
```

[4] We will use Kubernetes as our core distributed infrastructure. In fact we will use [k3d](https://k3d.io/v5.5.2/) which is a lightweight wrapper to run k3s (Rancher Lab’s minimal Kubernetes distribution) in docker

To install k3d:
```bash
wget -q -O - https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=v5.0.0 bash
k3d cluster create dist-ml
kubectl get nodes
```

[5] [kubectx](https://github.com/ahmetb/kubectx/) and kubens to easily switch contexts and namespaces
- `brew install kubectx`

[6] We will use [Kubeflow](https://www.kubeflow.org) to submit jobs to the Kubernetes cluster

![Kubeflow UI](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/aa731d8d-93cf-4089-a7a4-4a9b0f47e4eb "https://www.kubeflow.org/docs/started/architecture/")

[7] We wil also use [Argo workflows](https://argoproj.github.io/workflows) to construct and submit end to end machine learning workflows

For example if you want to create a kubernetes pod, then create a hello-world.yaml and then do `kubectl create -f hello-world.yaml` 

```yaml
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

Create a namespace using `kubectl create namespace kubeflow`

## Introduction

We are building an image classification end-to-end machine learning system. 

## System Architecture

<img width="1143" alt="Screenshot 2023-06-30 at 12 50 13 PM" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/18bb1322-1970-4ef4-a3a6-f7d345623ee0">

## Data Ingestion

We will use the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:

![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/9356978d-f3ab-4404-b35b-d5e50b3c82cb)

Here, 60,000 images are used to train the network and 10,000 images are used to evaluate how accurately the network learned to classify images.

### Single-node Data Pipeline

The `tf.data` API enables you to build complex input pipelines from simple, reusable pieces. It's very efficient. It makes it possible to handle large amounts of data, read from different data formats, and perform complex transformations.

Load the fashion-mnist dataset into a `tf.data.Dataset` object and do some preprocessing. We normalize the image pixel values from the [0, 255] range to the [0, 1] range. We are keeping an in-memory cache to improve performance. We also shuffle the training data.

```python
import tensorflow_datasets as tfds
import tensorflow as tf

def mnist_dataset():
    BUFFER_SIZE = 10000
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train = datasets['train']
    return mnist_train.map(scale).cache().shuffle(BUFFER_SIZE)
```

We have used the tensorflow_datasets module which contains a collection of datasets ready to use. This gives us a shuffled dataset where each element consists of images and labels.

### Distributed Data Pipeline

We can consume our dataset in a distributed fashion as well and to do that we can use the same function we created before with some tweaks. When training a model with multiple GPUs, you can use the extra computing power effectively by increasing the batch size. In general, use the largest batch size that fits the GPU memory.

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
```

The `num_replicas_in_sync` equals the number of devices that are used in the [all-reduce]() operation. We have used the `tf.distribute.MultiWorkerMirroredStrategy` API and with the help of this strategy, a Keras model that was designed to run on a single worker can seamlessly work on multiple workers with minimal code changes.

We have also enabled automatic data sharding across workers by setting `tf.data.experimental.AutoShardPolicy` to `AutoShardPolicy.DATA`. This setting is needed to ensure convergence and performance. Sharding means handing each worker a subset of the entire dataset. You can read more about it [here](https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions).

```python
with strategy.scope():
    dataset = mnist_dataset().batch(BATCH_SIZE).repeat()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    model = build_and_compile_cnn_model()

model.fit(dataset, epochs=3, steps_per_epoch=70)
```

## Model Training

Now we have created a data ingestion component for distributed ingestion and have enabled the sharding as well.

```python
def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
  
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
  return model
```

Next, we created a model and instantiated the optimizer. We are using accuracy to evaluate the model and sparse categorical cross entropy as the loss function.

Now we can train the model. We are also defining callbacks that will be executed during model training.

1. `tf.keras.callbacks.ModelCheckpoint` saves the model at a certain frequency, such as after every epoch

```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
```
Here we are defining the checkpoint directory to store the checkpoints and  the name of the checkpoint files.

2. `tf.keras.callbacks.TensorBoard` writes a log for TensorBoard, which allows you to visualize the graphs
3. `tf.keras.callbacks.LearningRateScheduler` schedules the learning rate to change after, for example, every epoch/batch

```python
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5
```

4. PrintLR prints the learning rate at the end of each epoch

```python
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(        epoch + 1, model.optimizer.lr.numpy()))
```

We put together all the callbacks.

```python
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]
```

Next, we can train the model

```python
single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(dataset, epochs=3, steps_per_epoch=70, callbacks=callbacks)
```

After training, we get an accuracy of 94% on the training data.

### Distributed Model Training

Next, we can insert the distributed training logic so that we can train the model on multiple workers. We are using the MultiWorkerMirroredStrategy with Keras.

In general, there are two common ways to do [distributed training with data parallelism](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras):
1. Synchronous training, where the steps of training are synced across the workers and replicas. Here, all workers train over different slices of input data in sync, and aggregating gradients at each step.
2. Asynchronous training, where the training steps are not strictly synced. Here, all workers are independently training over the input data and updating variables asynchronously. For instance, [parameter server training](https://www.tensorflow.org/tutorials/distribute/parameter_server_training).

We are using the MultiWorkerMirroredStrategy which implements synchronous distributed training across multiple workers, each with potentially multiple GPUs. It replicates all variables and computations to each local device and uses distributed collective implementation (e.g. all-reduce) so that multiple workers can work together.

Once we define our distributed training strategy, we initiate our distributed input data pipeline and the model inside the strategy scope.

### Model saving and loading

To save the model using `model.save`, the saving destinations need to be different for each worker.

- For non-chief workers, save the model to a temporary directory
- For the chief, save the model to the provided directory
The temporary directories of the workers need to be unique to prevent errors. The model saved in all the directories is identical, and only the model saved by the chief should be referenced for restoring or serving.

We will not save the model to temporary directories as it will waste our computing resources and memory. We will determine which worker is the chief and save its model only.
We can determine if the worker is the chief or not using the environment variable `TF_CONFIG`. Here's an example configuration:

```python
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
```
The `_is_chief` is a utility function that inspects the cluster spec and current task type and returns True if the worker is the chief and False otherwise.

```python
def _is_chief():
  return TASK_INDEX == 0

tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
TASK_INDEX = tf_config['task']['index']

if _is_chief():
    model_path = args.saved_model_dir
else:
    model_path = args.saved_model_dir + '/worker_tmp_' + str(TASK_INDEX)

multi_worker_model.save(model_path)
```

### Containerization

We put everything we wrote till now into a Python script called `multi-worker-distributed-training.py`. Now we can dockerize it to train the model in the Kubernetes cluster.

```dockerfile
FROM python:3.9
RUN pip install tensorflow==2.12.0 tensorflow_datasets==4.9.2
COPY multi-worker-distributed-training.py /
```

We then build the image from the dockerfile and import it to the k3d cluster as it does not have access to the image registry.

```bash
docker build -f Dockerfile -t kubeflow/multi-worker-strategy:v0.1 .
k3d image import kubeflow/multi-worker-strategy:v0.1 --cluster dist-ml
```

## Model Serving

## End-to-end Workflow
