import argparser
import torch
from torch import distributed

from dataset.transform import Compose, ToTensor, Normalize
from methods import get_method
from task import Task
from utils.logger import Logger

from PIL import Image
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparser.get_argparser()
    parser.add_argument("--image", type=str)

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    distributed.init_process_group(backend='nccl', init_method='env://')
    if opts.device is not None:
        device_id = opts.device
    else:
        device_id = opts.local_rank
    device = torch.device(device_id)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    if opts.device is not None:
        torch.cuda.set_device(opts.device)
    else:
        torch.cuda.set_device(device_id)
    opts.device_id = device_id
    opts.max_iter=1

    task = Task(opts)
    logger = Logger("./demo/", rank=rank)


    model = get_method(opts, task, device, logger)
    checkpoint = torch.load(opts.ckpt, map_location='cpu')
    model.model.eval()
    state = {}
    for k, v in checkpoint['model_state']['model'].items():
        state[k[7:]] = v
    model.model.load_state_dict(state)
    del checkpoint

    demo_transform = Compose([
            # PadCenterCrop(size=opts.crop_size_test),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(opts.image).convert('RGB')
    image = demo_transform(image).unsqueeze(0).to(model.device)

    outputs, _, _, _ = model.model(image)
    _, prediction = outputs.max(dim=1)
    prediction = prediction.cpu().numpy()

    plt.imsave(f"demo/{opts.image.split('/')[-1].split('.')[0]}.png", prediction.squeeze(), cmap='gray')



    