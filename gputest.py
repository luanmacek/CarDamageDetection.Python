import tensorflow as tf

# List physical devices (GPUs)
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) == 0:
    print("No GPUs available.")
else:
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"- {gpu.name}")
