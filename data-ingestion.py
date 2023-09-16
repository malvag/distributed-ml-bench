import tensorflow_datasets as tfds

datasets, info = tfds.load(name='fashion_mnist', with_info=True, as_supervised=True)
