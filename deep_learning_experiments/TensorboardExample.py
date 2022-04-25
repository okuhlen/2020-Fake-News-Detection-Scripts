import os
import threading

def run_tensorboard():
    os.system("tensorboard --logdir=C:\Project\MIT_2020_FND\data\cnn_dnn\log_20210320-122338")
    return

t = threading.Thread(target=run_tensorboard, args=([]))
t.start()