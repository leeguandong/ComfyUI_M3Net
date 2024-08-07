import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

from .M3Net import M3Net
from .data.custom_transforms import *

current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class M3Net_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "m3net": (
                    [
                        "M3Net-R", "M3Net-S"
                    ], {
                        "default": "M3Net-S"
                    }),
            }
        }

    RETURN_TYPES = ("M3NET",)
    RETURN_NAMES = ("m3net",)
    FUNCTION = "load_model"
    CATEGORY = "m3net"

    def load_model(self, m3net):
        if m3net == "M3Net-S":
            model = M3Net(embed_dim=512, dim=64, img_size=384, method="M3Net-S")
        else:
            model = M3Net(embed_dim=384, dim=64, img_size=384, method="M3Net-R")
        model_path = os.path.join(current_directory, "weights/M3Net-S.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return (model,)


class M3Net_Interface:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "m3net": ("M3NET",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "process"
    CATEGORY = "m3net"

    def process(self, m3net, image):
        processed_images = []
        processed_masks = []

        # import pdb;pdb.set_trace()
        for image in image:
            image_pil = tensor2pil(image).convert("RGB")
            comp = []
            comp.append(static_resize(size=[384, 384]))
            comp.append(tonumpy())
            comp.append(normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            comp.append(totensor())
            sample = {"image": image_pil, "shape": image_pil.size}
            data_batch = transforms.Compose(comp)(sample)

            images = data_batch['image']
            image_w, image_h = data_batch['shape']
            image_w, image_h = int(image_w), int(image_h)
            images = Variable(images.cuda()[None, :, :, :])

            outputs_saliency = m3net(images)
            mask_1_1 = outputs_saliency[-1]
            pred = torch.sigmoid(mask_1_1)

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_h, image_w))
            ])

            pred = pred.squeeze(0)
            pred = transform(pred)

            no_bg_image = Image.new("RGBA", pred.size, (0, 0, 0, 0))
            no_bg_image.paste(image_pil, mask=pred)

            new_im_tensor = pil2tensor(no_bg_image)
            pil_im_tensor = pil2tensor(pred)

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks


NODE_CLASS_MAPPINGS = {
    "M3Net_ModelLoader": M3Net_ModelLoader,
    "M3Net_Interface": M3Net_Interface
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "M3Net_ModelLoader": "M3Net Model Loader",
    "M3Net_Interface": "M3Net Interface"
}
