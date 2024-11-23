import argparse
import os
import lightning.pytorch as pl
import torch.cuda
import pdf2image
import sys

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Normalize
from torchvision.utils import save_image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pathlib import Path

# add top-folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from config import Config
from modules.xtra import XTRA
from utils import print_config
from augmentations import make_transforms


def list_files_in_directory(directory):
    base_path = Path(directory)
    return [str(path.relative_to(base_path)) for path in base_path.rglob('*') if path.is_file()]


def load_pil_image(image_path):
    # load PIL image
    if image_path.endswith('.pdf'):
        pil_image = pdf2image.convert_from_path(image_path, first_page=1, last_page=1)[0]
    else:
        pil_image = Image.open(image_path).convert('RGB')
    return pil_image


def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='path to pretrained models')
    parser.add_argument('--images', type=str, default=None,
                        help='path to directory of image to be used as input')
    parser.add_argument('--save_path', type=str, required=True,
                        help='directory to save output')
    parser.add_argument('--aim_idx', type=int, default=0,
                        help='AIM projection index. 0 is for next token prediction, 1 is for next-next and so on.')

    i_args = parser.parse_args()
    args = Config(i_args.config)
    if i_args.pretrained is not None: args.pretrained = i_args.pretrained
    if i_args.images is not None: args.images = i_args.images
    if i_args.save_path is not None: args.save_path = i_args.save_path
    if i_args.aim_idx is not None: args.aim_idx = i_args.aim_idx
    args.dataset = None  # don't load any datasets

    # print configuration
    print_config(args)

    # seed
    pl.seed_everything(args.seed)

    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # model
    model = XTRA(args).eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # get list of images
    image_names = list_files_in_directory(args.images)

    # get transform
    transform = make_transforms(**args.val_transform)

    # reverse transform
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32)
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32)
    unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    # inference loop
    for image_name in tqdm(image_names):
        # load PIL image
        img_path = os.path.join(args.images, image_name)
        pil_image = load_pil_image(img_path)

        # transform
        image = transform(pil_image)

        # save rescaled image
        rescaled_image_name = os.path.join(os.path.split(image_name)[0], 'rescaled_' + os.path.split(image_name)[1])
        rescaled_path = os.path.join(args.save_path, rescaled_image_name)
        os.makedirs(os.path.split(rescaled_path)[0], exist_ok=True)
        save_image(unnormalize(image), rescaled_path)

        if torch.cuda.is_available():
            image = image.cuda()

        # run encoder
        pred, target = model(image.unsqueeze(0), aim_idx=args.aim_idx, return_target=True)
        pred, target = pred[0], target[0]

        # shift to align with target
        pred[1 + args.aim_idx:] = pred[:-1 - args.aim_idx].clone()
        pred[:1 + args.aim_idx] = 0

        # get target crops
        target_shape = target.shape

        # inverse pix norm if necessary
        if args.aim_norm_pix_loss:
            target = target.flatten(1)  # flatten to get mean and variance
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred = pred * (var + 1.0e-6) ** .5 + mean

        # convert back to RGB image
        img = pred.reshape(target_shape)                         # (num_crops, 3, crop_size, crop_size)
        img = torch.movedim(img, 1, 3)                           # (num_crops, crop_size, crop_size, 3)
        h = w = int(model.enc.img_size[0] / model.enc.crop_size)
        img = model.decropify_sample(img, h=h, w=w)              # (img_height, img_width, 3)

        # inverse transform
        img = unnormalize(img)

        # save reconstruction
        save_image(img, os.path.join(args.save_path, image_name))


if __name__ == '__main__':
    main()