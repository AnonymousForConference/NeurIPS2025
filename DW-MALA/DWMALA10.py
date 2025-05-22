import random
from VGG import build_vgg_model
from dataloader.CIFAR10 import load_cifar10
from active_learning_loop import active_learning_loop
from scipy.spatial.distance import cdist

print('cifar-10 start')



(x_train, y_train), (x_test, y_test) = load_cifar10()
model = build_vgg_model()
dwmala_train_acc, dwmala_test_acc = active_learning_loop(model, x_train, y_train, x_test, y_test)

