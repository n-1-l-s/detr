import torchvision
import torch
import argparse


from matplotlib import pyplot as plt
from dataset.doclaynet import build
from models import build_model
from main import get_args_parser
from PIL import Image
import dataset.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.patches as patches


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
args.dataset_file = "doclaynet"
args.device = "cpu"

#ds = build("val", args)
model, criterion, postprocessors = build_model(args)

#checkpoint = torch.load("/lustre/hpc_home/adb614s/doc_analysis/run_output/checkpoint.pth", map_location=torch.device("cpu"), weights_only=False)
checkpoint = torch.load("./checkpoint0299.pth", map_location=torch.device("cpu"), weights_only=False)
model.load_state_dict(checkpoint['model'])
model.to(args.device)
model.eval()

img = Image.open("ahlen.jpg")

w, h = img.size
print(img.size)
img_orig = img

normalize = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


img = normalize(img, None)[0]
out = model(img.unsqueeze(0))


out["pred_boxes"][:,:,0] *= w
out["pred_boxes"][:,:,1] *= h
out["pred_boxes"][:,:,2] *= w 
out["pred_boxes"][:,:,3] *= h

out["pred_boxes"][:,:,0] -= out["pred_boxes"][:,:,2] / 2
out["pred_boxes"][:,:,1] -= out["pred_boxes"][:,:,3] / 2

print(out["pred_boxes"][0][0])

classes = np.argmax(out["pred_logits"][0].detach().numpy(), axis=1)


fig, ax = plt.subplots()
ax.imshow(img_orig)
for i in range(100):
    if classes[i] != 20 and out["pred_logits"][0][i][classes[i]] > 0:
        box = out["pred_boxes"][0][i].detach().numpy()
        rect = patches.Rectangle(box[:2], box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
plt.show()

#result = torchvision.utils.draw_bounding_boxes(T.ToTensor()(img_orig, None)[0], out["pred_boxes"][0], width=5)
#show(result)

#print(ds[0][1]["labels"])

#print(model)
#print(ds[0][0].shape)
#print(out)
