from CNNClassifier import CNNClassifier

cnn = CNNClassifier()
cnn.restore_model(print_accuracy=False)

# cnn.generate_adversarial_examples(fast_sign=False, source_target=False, step_size=1./255.)
# cnn.generate_adversarial_examples(fast_sign=False, source_target=True, step_size=1./255.)
# cnn.generate_adversarial_examples(fast_sign=True, source_target=False, step_size=1./255.)
# cnn.generate_adversarial_examples(fast_sign=True, source_target=True, step_size=1./255.)
#
# cnn.generate_adversarial_examples(fast_sign=False, source_target=False, step_size=10./255.)
# cnn.generate_adversarial_examples(fast_sign=False, source_target=True, step_size=10./255.)
# cnn.generate_adversarial_examples(fast_sign=True, source_target=False, step_size=10./255.)
# cnn.generate_adversarial_examples(fast_sign=True, source_target=True, step_size=10./255.)

cls_target = False
# cnn.generate_general_adversarial_examples(step_size=(.5 / 255.0), fast_sign=True, cls_target=cls_target)
# cnn.generate_general_adversarial_examples(step_size=(1 / 255.0), fast_sign=True, cls_target=cls_target)
# cnn.generate_general_adversarial_examples(step_size=(5 / 255.0), fast_sign=True, cls_target=cls_target)
# cnn.generate_general_adversarial_examples(step_size=(10 / 255.0), fast_sign=True, cls_target=cls_target)
# cnn.generate_general_adversarial_examples(step_size=(40 / 255.0), fast_sign=True, cls_target=cls_target)

# cnn.generate_class_adversarial_examples(step_size=(.5 / 255.), epochs=20, noise_limit=.2, fast_sign=False,
#                                         cls_target=False)  # 56%
# cnn.generate_class_adversarial_examples(step_size=(.5 / 255.), noise_limit=.225, fast_sign=False,
#                                         cls_target=False)  # 68%
# cnn.generate_class_adversarial_examples(step_size=(2./255.), noise_limit=.25, fast_sign=False, cls_target=False)  # 75%
# cnn.generate_class_adversarial_examples(step_size=(2./255.), noise_limit=.275, fast_sign=False, cls_target=False)  # ?%

noise_limits = [.15, .16, .17, .18, .19, .20, .21, .22, .23, .24, .25, .26, .27, .28, .29, .30]
step_sizes = [.25/255., .5/255., .75/255., 1.0/255., 1.25]

for noise_limit in noise_limits:
    for step_size in step_sizes:
        cnn.generate_class_adversarial_examples(step_size=step_size, epochs=30, noise_limit=noise_limit, fast_sign=False, cls_target=False)
