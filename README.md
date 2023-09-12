# Distributed ML Systems

Recently, I was involved in a classification-based ML project where we developed a distributed scalable ML service. So I wanted to build it again to deepen my understanding.

Why distributed systems? Distributing machine learning systems allows developers to handle extremely large datasets across multiple clusters, take advantage of automation tools, and benefit from hardware accelerations. This repo includes code and references to implement a scalable and reliable machine learning system.

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

[4] We will use Kubernetes as our core distributed infrastructure. In fact, we will use [k3d](https://k3d.io/v5.5.2/) which is a lightweight wrapper to run k3s (Rancher Lab’s minimal Kubernetes distribution) in docker. Its great for local Kubernetes development.

To install k3d:
```bash
wget -q -O - https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=v5.0.0 bash
k3d cluster create dist-ml
kubectl get nodes
```

<img width="794" alt="image" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/d9345b11-6fce-49f0-8ef6-36a999840d0c">


[5] [kubectx](https://github.com/ahmetb/kubectx/) and kubens to easily switch contexts and namespaces
- `brew install kubectx`

[6] We will use [Kubeflow](https://www.kubeflow.org) to submit jobs to the Kubernetes cluster

![Kubeflow UI](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/aa731d8d-93cf-4089-a7a4-4a9b0f47e4eb "https://www.kubeflow.org/docs/started/architecture/")

[7] We wil also use [Argo workflows](https://argoproj.github.io/workflows) to construct and submit end to end machine learning workflows

For example if you want to create a kubernetes pod, then create a hello-world.yaml as below.

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

Next, submit the job to our cluster

```bash
kubectl create -f hello-world.yaml
```

We can see the statuses

```bash
kubectl get pods
```

We can see what is being printed out in the container

```bash
kubectl logs whalesay
```

If you want to get the details of a single pod with the raw yaml, then enter the following command

```bash
kubectl get pod whalesay -o yaml
```

You can get the JSON or any other format as well.

Next, we create a namespace. The namespaces provide a mechanism for isolating groups of resources within a single cluster.

To create a namespace

```bash
kubectl create namespace kubeflow
```

Next, switch to kubeflow

```bash
kubens kubeflow
```

<img width="603" alt="image" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/54f180ee-bc0a-4e8f-873f-0be8ed5cbbe8">


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
  print("Training CNN model")
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

We then build the image from the dockerfile.

```bash
docker build -f Dockerfile -t kubeflow/multi-worker-strategy:v0.1 .
```

<img width="865" alt="image" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/f4a6fbdb-0704-4f61-963d-b876874a2183">

Next, import the above image to the k3d cluster as it cannot access the image registry.

```bash
k3d image import kubeflow/multi-worker-strategy:v0.1 --cluster dist-ml
```

<img width="861" alt="image" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/c96adade-741e-4bd6-a958-49007917a844">

Now when the pods are completed/failed, all files in the pods are recycled by the Kubernetes garbage collection. So all the model checkpoints are lost and we don't have a model for serving. To avoid this we use PersistentVolume(PV) and PersistentVolumeClaim(PVC).

A PersistentVolume (PV) is a piece of storage in the cluster that has been provisioned by an administrator or dynamically provisioned. It is a resource in the cluster just like a node is a cluster resource. PVs are volume plugins like Volumes but have a lifecycle independent of any individual Pod that uses the PV. This means that PV will persist and live even when the pods are deleted.

A PersistentVolumeClaim (PVC) is a request for storage by a user. It is similar to a Pod. Pods consume node resources and PVCs consume PV resources. Pods can request specific levels of resources (CPU and Memory). Claims can request specific size and access modes (e.g., they can be mounted ReadWriteOnce, ReadOnlyMany or ReadWriteMany).

We can create a PVC to submit a request for storage that will be used in worker pods to store the trained model. Here we are requesting 1GB storage with ReadWriteOnce mode.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: volume
spec:
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  resources:
    requests:
      storage: 1Gi
```

Next, we create the PVC.

```bash
kubectl create -f multi-worker-pvc.yaml
```

Next, we will define a TFJob specification with the image we built before that contains the distributed training script.

```yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: multi-worker-training
spec:
  runPolicy:
    cleanPodPolicy: None
  tfReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: tensorflow
              image: kubeflow/multi-worker-strategy:v0.1
              imagePullPolicy: IfNotPresent
              command: ["python", "/multi-worker-distributed-training.py", "--saved_model_dir", "/trained_model/saved_model_versions/2/", "--checkpoint_dir", "/trained_model/checkpoint"]
              volumeMounts:
                - mountPath: /trained_model
                  name: training
              resources:
                limits:
                  cpu: 500m
          volumes:
            - name: training
              persistentVolumeClaim:
                claimName: volume
```

We pass the arguments (`saved_model_dir`, `checkpoint_dir`) to the container. The `volumes` field specifies the persistent volume claim and `volumeMounts` field specifies what folder to mount the files. The `CleanPodPolicy` in the TFJob spec controls the deletion of pods when a job terminates. The `restartPolicy` determines whether pods will be restarted when they exit.

Next, we submit this TFJob to our cluster and start our distributed model training.

```bash
kubectl create -f multi-worker-tfjob.yaml
```

Let's start the pods and train our distributed model. We can see the logs from the pods below.



## Model Selection

We've implemented the distributed model training component. In production, we might need to train different models and pick the top performer for model serving. Let's create two more models to understand this concept.

```python
def build_and_compile_cnn_model_with_batch_norm():
  print("Training CNN model with batch normalization")
  model = models.Sequential()
  model.add(layers.Input(shape=(28, 28, 1), name='image_bytes'))
  model.add(layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('sigmoid'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('sigmoid'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))

  model.summary()

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model
```

```python
def build_and_compile_cnn_model_with_dropout():
  print("Training CNN model with dropout")
  model = models.Sequential()
  model.add(layers.Input(shape=(28, 28, 1), name='image_bytes'))
  model.add(layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))

  model.summary()

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model
```

We train these models by submitting three different TFJobs with an argument `--model_type`.

Next, we load the testing data and the trained model to evaluate its performance. The model with the highest accuracy score can be moved to a different folder and used for model serving.

```python
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

best_model_path = ""
best_accuracy = 0

for i in range(3):
  model_path = "trained_models/saved_model_versions/" + str(i)
  model = tf.keras.models.load_model(model_path)

  datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
  mnist_test = datasets['test']
  ds = mnist_test.map(scale).cache().shuffle(BUFFER_SIZE).batch(64)
  loss, accuracy = model.evaluate(ds)

  if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_model_path = model_path

dst = "trained_model/saved_model_versions/3"
shutil.copytree(best_model_path, dst)
```

We add this script to the Dockerfilw, rebuild the image, and create a pod that runs model serving.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-selection
spec:
  containers:
  - name: predict
    image: kubeflow/multi-worker-strategy:v0.1
    command: ["python", "/model-selection.py"]
    volumeMounts:
    - name: model
      mountPath: /trained_model
  volumes:
  - name: model
    persistentVolumeClaim:
      claimName: strategy-volume
```

## Model Serving

We implemented distributed training and model selection. Now we implement the model serving component. Here we take the trained model from `trained_model/saved_model_versions/3`. The model serving should be very performant.

### Single server model inference

```python
model_path = "trained_models/saved_model_versions/3"
model = tf.keras.models.load_model(model_path)
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_test = datasets['test']
ds = mnist_test.map(scale).cache().shuffle(BUFFER_SIZE).batch(64)
loss, accuracy = model.predict(ds)
```

## End-to-end Workflow
