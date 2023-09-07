import argparse
import json
import os

import tensorflow_datasets as tfds
import tensorflow as tf


def mnist_dataset():
    BUFFER_SIZE = 10000

    # Scale the MNIST data from [0, 255] range to [0, 1] range
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    
    # Use Fashion-MNIST: https://www.tensorflow.org/datasets/catalog/fashion_mnist
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train = datasets['train']

    return mnist_train.map(scale).cache().shuffle(BUFFER_SIZE)


def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5


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


def main(args):
   
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    with strategy.scope():
        dataset = mnist_dataset().batch(BATCH_SIZE).repeat()

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)

        multi_worker_model = build_and_compile_cnn_model()

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = args.checkpoint_dir

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {}'.format(        epoch + 1, multi_worker_model.optimizer.lr.numpy()))

    callbacks = [
      tf.keras.callbacks.TensorBoard(log_dir='./logs'),
      tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                         save_weights_only=True),
      tf.keras.callbacks.LearningRateScheduler(decay),
      PrintLR()
    ]

    multi_worker_model.fit(dataset,
                           epochs=1,
                           steps_per_epoch=70,
                           callbacks=callbacks)
    
    def _is_chief():
        return TASK_INDEX == 0

    if _is_chief():
        model_path = args.saved_model_dir

    else:
        # Save to a path that is unique across workers.
        model_path = args.saved_model_dir + '/worker_tmp_' + str(TASK_INDEX)

    multi_worker_model.save(model_path)


if __name__ == '__main__':

  tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
  TASK_INDEX = tf_config['task']['index']

  parser = argparse.ArgumentParser()
  parser.add_argument('--saved_model_dir',
                      type=str,
                      required=True,
                      help='Tensorflow export directory')

  parser.add_argument('--checkpoint_dir',
                      type=str,
                      required=True,
                      help='Tensorflow checkpoint directory')

  parsed_args = parser.parse_args()
  main(parsed_args)