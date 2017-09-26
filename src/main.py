from CNNClassifier import CNNClassifier

cnn = CNNClassifier()
cnn.restore_model()
cnn.generate_adversarial_examples(fast_sign=False, source_target=False, step_size=1./255.)
cnn.generate_adversarial_examples(fast_sign=False, source_target=True, step_size=1./255.)
cnn.generate_adversarial_examples(fast_sign=True, source_target=False, step_size=1./255.)
cnn.generate_adversarial_examples(fast_sign=True, source_target=True, step_size=1./255.)

cnn.generate_adversarial_examples(fast_sign=False, source_target=False, step_size=10./255.)
cnn.generate_adversarial_examples(fast_sign=False, source_target=True, step_size=10./255.)
cnn.generate_adversarial_examples(fast_sign=True, source_target=False, step_size=10./255.)
cnn.generate_adversarial_examples(fast_sign=True, source_target=True, step_size=10./255.)

