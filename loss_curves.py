import matplotlib.pyplot as plt
def plot_loss_curves(results):
  plt.figure(figsize=(10,7))
  plt.subplot(1,2,1)
  plt.plot(results["test_loss"],label="testing loss")
  plt.plot(results["train_loss"],label="training loss")
  plt.legend()
  plt.subplot(1,2,2)
  plt.plot(results["test_acc"],label="testing accuracy")
  plt.plot(results["train_acc"],label="training accuracy")
  plt.legend()
