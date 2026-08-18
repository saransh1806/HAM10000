from PIL import Image
import torch
import matplotlib.pyplot as plt
def pred_and_plot(model,
                  img,
                  class_names,
                  device):
  model.to(device)
  model.eval()
  with torch.inference_mode():
    transformed_img=(img).unsqueeze(dim=0).to(device)
    y_logits=model(transformed_img)
    y_prob=torch.softmax(y_logits,dim=1)
    y_label=torch.argmax(y_prob,dim=1).item()
    plt.figure()
    plt.imshow(img.permute(1,2,0))
    plt.title(f"Pred {class_names[y_label]} | Pred prob {y_prob.max().item():.8f}")
    plt.axis(False)
  return y_prob

