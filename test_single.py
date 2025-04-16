import numpy as np
from PIL import Image
import cv2, torch, fractions
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from models.projected_model import fsModel


transformer = transforms.Compose([transforms.ToTensor()])
transformer_Arcface = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)


if __name__ == '__main__':
    opt = TestOptions().parse()
    torch.nn.Module.dump_patches = True

    model = fsModel()
    model.initialize(opt)
    model.eval()

    with torch.no_grad():
        pic_a = opt.pic_a_path                                                             # source face image
        img_a = Image.open(pic_a).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        pic_b = opt.pic_b_path                                                             # target face image
        img_b = Image.open(pic_b).convert('RGB')
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        img_id = img_id.cuda()                                                             # convert numpy to tensor
        img_att = img_att.cuda()

        latend_id = model.netArc(F.interpolate(img_id, size=(112, 112))).detach().to('cpu')# create latent id
        latend_id_nor = F.normalize(latend_id, p=2, dim=1).to('cuda')

        # forward process
        img_fake = model.netG(img_att, latend_id_nor)

        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        row1 = row1.detach().to('cpu') * imagenet_std + imagenet_mean
        row2 = row2.detach().to('cpu')
        row3 = row3.detach().to('cpu') * imagenet_std + imagenet_mean
        full = np.array(torch.cat([row1, row2, row3], dim=2).permute(1, 2, 0))             # C * W * H -> W * H * C
        output = full[..., ::-1] * 255

        cv2.imwrite(opt.output_path + 'result.jpg', output)