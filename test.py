import argparse
import os
import random
import time
import cv2
import glob
import torch
import numpy as np
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from natsort import natsorted
from basicsr.models.archs.modem_arch import Modem


def main():
    print('Loading...')
    model = Modem(
        n_feats=36,
        n_encoder_res=[1, 1],
        n_sr_blocks=[4, 4, 6, 8, 6, 4, 4],
    ).cuda()
    print(model)
    print(f'loading model from {opt.ckpt_path}')
    model.load_state_dict(torch.load(opt.ckpt_path, map_location='cpu')['params'], strict=True)
    model.eval()
    psnr_list = []
    ssim_list = []

    for idx, path in enumerate(natsorted(glob.glob(os.path.join(opt.input_folder, '*')))):
        img_name = os.path.basename(path)
        gt_path = os.path.join(opt.gt_folder, img_name)
        print(f'Processing {idx + 1}/{len(os.listdir(opt.input_folder))}: {img_name}')
        # read image
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).cuda()  # CHW-RGB to NCHW-RGB

        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = 8 - (h_old % 8) if h_old % 8 != 0 else 0
            w_pad = 8 - (w_old % 8) if w_old % 8 != 0 else 0
            img_lq = torch.nn.functional.pad(img_lq, (0, w_pad, 0, h_pad), mode='reflect')

            out = model(img_lq)
            img_clean = out[..., :h_old, :w_old]

        # save image
        output = img_clean.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.uint8)
        psnr = calculate_psnr(gt_img, output, crop_border=0, test_y_channel=True)

        ssim = calculate_ssim(gt_img, output, crop_border=0, test_y_channel=True)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(f'PSNR: {psnr:.2f}, SSIM: {ssim:.4f}')
        cv2.imwrite(os.path.join(opt.output_folder, img_name), output, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f'Processing done! All images are saved in the {opt.output_folder}.')
    print(f'Average PSNR: {np.mean(psnr_list):.4f}, Average SSIM: {np.mean(ssim_list):.5f}')


def set_seed(seed=100):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder')
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to the ground truth folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')
    opt = parser.parse_args()
    set_seed()
    # Check if the input folder exists
    if os.path.exists(opt.output_folder):
        os.rename(opt.output_folder, opt.output_folder + time.strftime("%Y%m%d%H%M%S"))
        print(f"Output folder already exists. Renamed to {opt.output_folder + time.strftime('%Y%m%d%H%M%S')}")
    os.makedirs(opt.output_folder, exist_ok=True)
    print(f"Output folder created: {opt.output_folder}")
    # Run the main function
    main()
