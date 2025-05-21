from VGG import build_vgg_model
from dataloader.tiny import load_tiny
from active_learning_loop import active_learning_loop

print('imagenet start')

(x_train, y_train), (x_test, y_test) = load_tiny()
model = build_vgg_model()
dwmala_train_acc, dwmala_test_acc = active_learning_loop(model, x_train, y_train, x_test, y_test)
