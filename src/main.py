from CNNClassifier import CNNClassifier
import time
import numpy as np
cnn = CNNClassifier()
cnn.restore_model(print_accuracy=False)

# # cnn.generate_adversarial_examples(fast_sign=False, source_target=False, step_size=1./255.)
# # cnn.generate_adversarial_examples(fast_sign=False, source_target=True, step_size=1./255.)
# # cnn.generate_adversarial_examples(fast_sign=True, source_target=False, step_size=1./255.)
# # cnn.generate_adversarial_examples(fast_sign=True, source_target=True, step_size=1./255.)
# #
# # cnn.generate_adversarial_examples(fast_sign=False, source_target=False, step_size=10./255.)
# # cnn.generate_adversarial_examples(fast_sign=False, source_target=True, step_size=10./255.)
# # cnn.generate_adversarial_examples(fast_sign=True, source_target=False, step_size=10./255.)
# # cnn.generate_adversarial_examples(fast_sign=True, source_target=True, step_size=10./255.)
#
# cls_target = False
# # cnn.generate_general_adversarial_examples(step_size=(.5 / 255.0), fast_sign=True, cls_target=cls_target)
# # cnn.generate_general_adversarial_examples(step_size=(1 / 255.0), fast_sign=True, cls_target=cls_target)
# # cnn.generate_general_adversarial_examples(step_size=(5 / 255.0), fast_sign=True, cls_target=cls_target)
# # cnn.generate_general_adversarial_examples(step_size=(10 / 255.0), fast_sign=True, cls_target=cls_target)
# # cnn.generate_general_adversarial_examples(step_size=(40 / 255.0), fast_sign=True, cls_target=cls_target)
#
# # cnn.generate_class_adversarial_examples(step_size=(.5 / 255.), epochs=20, noise_limit=.2, fast_sign=False,
# #                                         cls_target=False)  # 56%
# # cnn.generate_class_adversarial_examples(step_size=(.5 / 255.), noise_limit=.225, fast_sign=False,
# #                                         cls_target=False)  # 68%
# # cnn.generate_class_adversarial_examples(step_size=(2./255.), noise_limit=.25, fast_sign=False, cls_target=False)  # 75%
# # cnn.generate_class_adversarial_examples(step_size=(2./255.), noise_limit=.275, fast_sign=False, cls_target=False)  # ?%
#

if True:
    noise_limits = [.24]
    epochs = [25]
    step_sizes = [.5/255.]
    src_targets = [3]

    for noise_limit in noise_limits:
        for step_size in step_sizes:
            for epoch in epochs:
                for src_target in src_targets:
                    duration = time.time()
                    cnn.generate_class_adversarial_examples(
                        src_target=src_target,
                        step_size=step_size,
                        epochs=epoch,
                        noise_limit=noise_limit,
                        fast_sign=False,
                        cls_target=False
                    )
                    print('time: %s' % (time.time()-duration))

# cnn.generate_adversarial_examples(noise_limit=.15, step_size=(5./255.))
