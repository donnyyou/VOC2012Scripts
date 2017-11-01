import torch
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from transform import Colorize
from torchvision.transforms import ToPILImage, Compose, ToTensor, CenterCrop, Normalize
from transform import Scale, ToLabel, ReLabel, CocoLabel
from model import FCN
from datasets import VOCTestSet
from PIL import Image
import numpy as np

import config

import cv2
import os
import shutil
shutil.rmtree("./test")
os.makedirs("./test")
from evaluation import get_iou_list


image_transform = Compose([Scale((256, 256), Image.BILINEAR), 
                           ToTensor(),
                           Normalize([.485, .456, .406], [.229, .224, .225]),])

target_transform = Compose([
    Scale((256, 256), Image.NEAREST),
    ToLabel(),
    CocoLabel(),
])

batch_size = 1
dst = VOCTestSet("/root/group-incubation-bj", img_transform=image_transform, label_transform=target_transform)

testloader = data.DataLoader(dst, batch_size=batch_size,
                             num_workers=8)

model = torch.nn.DataParallel(FCN(config.NUM_CLASSES))
model.cuda()
model.load_state_dict(torch.load(config.TORCH_MODEL_PATH))
model.eval()

valid_image_count = list()
iou_score = list()
line_count = 0
for i in range(config.NUM_CLASSES):
    valid_image_count.append(0)
    iou_score.append(0.0)

for j, data in enumerate(testloader):
    imgs, labels, names, original_size = data
    width = list(original_size)[0][0]
    height = list(original_size)[1][0]

    imgs = Variable(imgs.cuda())
    outputs = model(imgs)
    # 22 256 256
    filename = list(names)[0]
    for i, output in enumerate(outputs):
        output = output.data.max(0)[1]

        result = np.copy(output[0].cpu().numpy())
        # result = result.reshape((512, 1024))
        label = labels[0].numpy()
        iou_list = get_iou_list(result, label, config.NUM_CLASSES)
        for index in range(config.NUM_CLASSES):
            if iou_list[index] == -0.1:
                print "*****"
                continue
            valid_image_count[index] = valid_image_count[index] + 1
            iou_score[index] += iou_list[index]
        print line_count
        print iou_list
        line_count = line_count + 1
        output = Colorize()(output)
        output = np.transpose(output.numpy(), (1, 2, 0))
        img = Image.fromarray(output, "RGB")
        if i != 0:
            img = img.resize((width, height), Image.NEAREST)
        img.save("test/" + filename)

average = 0.0
for i in range(config.NUM_CLASSES):
    print "class %d: %f" % (i, iou_score[i]/max(valid_image_count[i], 1))
    if i != 0:
        average += iou_score[i]/max(valid_image_count[i], 1)

print "average: %f" % (average/5)
