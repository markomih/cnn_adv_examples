from CNNClassifier import CNNClassifier
import time


cnn = CNNClassifier()
cnn.restore_model(print_accuracy=False)



noise_limits = [.24]
epochs = [25]
step_sizes = [.5/255.]

for noise_limit in noise_limits:
    for step_size in step_sizes:
        for epoch in epochs:
            duration = time.time()
            cnn.generate_common_adversarial_noise(
                step_size=step_size,
                epochs=epoch,
                noise_limit=noise_limit,
                fast_sign=False,
                cls_target=False
            )
            print('time: %s' % (time.time()-duration))