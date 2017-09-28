from CNNClassifier import CNNClassifier

cnn = CNNClassifier()
cnn.restore_model()
# cnn.generate_adversarial_examples(fast_sign=False, source_target=False, step_size=1./255.)
# cnn.generate_adversarial_examples(fast_sign=False, source_target=True, step_size=1./255.)
# cnn.generate_adversarial_examples(fast_sign=True, source_target=False, step_size=1./255.)
# cnn.generate_adversarial_examples(fast_sign=True, source_target=True, step_size=1./255.)
#
# cnn.generate_adversarial_examples(fast_sign=False, source_target=False, step_size=10./255.)
# cnn.generate_adversarial_examples(fast_sign=False, source_target=True, step_size=10./255.)
# cnn.generate_adversarial_examples(fast_sign=True, source_target=False, step_size=10./255.)
# cnn.generate_adversarial_examples(fast_sign=True, source_target=True, step_size=10./255.)

cls_target = True
cnn.generate_general_adversarial_examples(step_size=(1 / 255.0), fast_sign=True, cls_target=cls_target)
cnn.generate_general_adversarial_examples(step_size=(5 / 255.0), fast_sign=True, cls_target=cls_target)
cnn.generate_general_adversarial_examples(step_size=(10 / 255.0), fast_sign=True, cls_target=cls_target)
