# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import lpips
import matplotlib.pyplot as plt
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from dataset import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models import WarpFieldVAE, DeepAppearanceVAE
from utils import Renderer, gammaCorrect, str2bool
import cv2
import numpy as np
import os
import argparse
import time
import torch.nn.functional as F
import json
import argparse
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from qlib.qwrap import model2ptq
from qlib.ptq import ActQuant
from qlib.base import QConvTranspose2dWN, QLinearWN
from qlib.utils import AverageMeter

from psnr import eval_errors

count = 0

def main(args, camera_config, test_segment):
    global count
    local_rank = os.environ['LOCAL_RANK']
    local_rank = int(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    dataset_train = Dataset(args.data_dir, args.krt_dir, args.framelist_test, args.tex_size,
                        camset=None if camera_config is None else camera_config['train'],
                        exclude_prefix=test_segment)

    dataset_test = Dataset(args.data_dir, args.krt_dir, args.framelist_test, args.tex_size,
                            camset=None if camera_config is None else camera_config['test'],
                            valid_prefix=test_segment)


    dataset_visual = Dataset(args.data_dir, args.krt_dir, args.framelist_test, args.tex_size,
                            camset=None if camera_config is None else [args.view_id],
                            valid_prefix=test_segment)
    
    visual_sampler = DistributedSampler(dataset_visual)

    visual_loader = DataLoader(dataset_visual, args.val_batch_size, sampler=visual_sampler, num_workers=args.n_worker)

    if local_rank == 0:
        print('#visual samples', len(dataset_visual))

    n_cams = len(set(dataset_train.cameras).union(set(dataset_test.cameras)))
    if args.arch == 'base':
        model = DeepAppearanceVAE(args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams).to(device)
    elif args.arch == 'res':
        model = DeepAppearanceVAE(args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, res=True, n_cams=n_cams).to(device)
    elif args.arch == 'warp':
        model = WarpFieldVAE(args.tex_size, args.mesh_inp_size, z_dim=args.nlatent, n_cams=n_cams).to(device)
    elif args.arch == 'non':
        model = DeepAppearanceVAE(args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, res=False, non=True, n_cams=n_cams).to(device)
    elif args.arch == 'bilinear':
        model = DeepAppearanceVAE(args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, res=False, non=False, bilinear=True, n_cams=n_cams).to(device)
    else:
        raise NotImplementedError
    
    tex_dec = model.dec
    qtex_dec = model2ptq(tex_dec)

    # total param size
    total_w = 0
    for m in qtex_dec.modules():
        if isinstance(m, (QConvTranspose2dWN, QLinearWN)):
            total_w += m.weight.numel()

    print(f"Total # of parameters = {total_w}")
    # modules = dict(qtex_dec.named_modules(remove_duplicate=False))
    if args.wbit < 32:

        count = 0
        insize = 4
        for name, layer in qtex_dec.named_modules():
            if isinstance(layer, QConvTranspose2dWN):
                layer.xq =  ActQuant(nbit=args.abit, gp=args.groupsize, sym= args.sym, cr= args.clip_ratio)
                if args.omni:

                        layer.set_smooth(alpha=0.2)
                insize = insize*2
    
    model.dec = qtex_dec
    
    # by default load the best_model.pth
    print('loading model from', args.model_path)
    print("View ID: ", args.view_id)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    state_dict = torch.load(args.model_path, map_location=map_location)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  

        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    
    model = model.to(device)
    print("Checkpoint loaded!")

    model = torch.nn.parallel.DistributedDataParallel(model, [local_rank], local_rank)
    renderer = Renderer()

    mse = nn.MSELoss()

    texmean = cv2.resize(dataset_test.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_test.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_test.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = dataset_test.texstd
    vertmean = torch.tensor(dataset_test.vertmean, dtype=torch.float32).view((1, -1, 3)).to(device)
    vertstd = dataset_test.vertstd
    loss_weight_mask = cv2.flip(cv2.imread(args.loss_weight_mask), 0)
    loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
    loss_weight_mask = torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to(device)

    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.video_path, exist_ok=True)
    os.makedirs(args.err_path, exist_ok=True)
    os.makedirs(args.tensor_path, exist_ok=True)

    def run_net(data):
        global count
        M = data['M'].cuda()
        gt_tex = data['tex'].cuda()
        vert_ids = data['vert_ids'].cuda()
        uvs = data['uvs'].cuda()
        uv_ids = data['uv_ids'].cuda()
        avg_tex = data['avg_tex'].cuda()
        view = data['view'].cuda()
        transf = data['transf'].cuda()
        verts = data['aligned_verts'].cuda()
        photo = data['photo'].cuda()
        mask = data['mask'].cuda()
        print("mask", mask.shape)
        cams = data['cam'].cuda()
        batch, channel, height, width = avg_tex.shape

        output = {}

        if args.arch == 'warp':
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    _ = model(avg_tex, verts, view, cams=cams)
            
            with open(os.path.join(args.result_path, 'time_%s.txt' % count), "w") as f:
                f.write(prof.key_averages().table(sort_by="self_cuda_time_total"))

            count+=1
            pred_tex, pred_verts, unwarped_tex, warp_field, kl = model(avg_tex, verts, view, cams=cams)
            output['unwarped_tex'] = unwarped_tex
            output['warp_field'] = warp_field
        else:
            pred_tex, pred_verts, kl = model(avg_tex, verts, view, cams=cams)
        vert_loss = mse(pred_verts, verts)

        pred_verts = pred_verts * vertstd + vertmean
        pred_tex = (pred_tex * texstd + texmean) / 255.
        gt_tex = (gt_tex * texstd + texmean) / 255.

        loss_mask = loss_weight_mask.repeat(batch, 1, 1, 1)
        tex_loss = mse(pred_tex * mask, gt_tex * mask) * (255**2) / (texstd**2)

        if args.lambda_screen > 0:
            screen_mask, rast_out = renderer.render(M, pred_verts, vert_ids, uvs, uv_ids, loss_mask, args.resolution)
            pred_screen, rast_out = renderer.render(M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, args.resolution)
            screen_loss = torch.mean((pred_screen - photo)**2 * screen_mask) * (255**2) / (texstd**2)
            psnr, ssim, _ = eval_errors(pred_screen, photo)
            print( psnr, ssim)
            psnr_m, ssim_m, _ = eval_errors(pred_screen* screen_mask, photo* screen_mask)
            print("PSNR: ", psnr_m)
            print("SSIM: ", ssim_m)
            data['screen_mask'] = screen_mask
        else:
            screen_loss, pred_screen = torch.zeros([]), None

        total_loss = 0
        if args.lambda_verts > 0:
            total_loss = total_loss + args.lambda_verts * vert_loss
        if args.lambda_tex > 0:
            total_loss = total_loss + args.lambda_tex * tex_loss
        if args.lambda_screen > 0:
            total_loss = total_loss + args.lambda_screen * screen_loss
        if args.lambda_kl > 0:
            total_loss = total_loss + args.lambda_kl * kl

        losses = {
            'total_loss': total_loss,
            'vert_loss': vert_loss,
            'screen_loss': screen_loss,
            'tex_loss': tex_loss,
            'kl': kl
        }

        output['gt_tex'] = gt_tex
        output['pred_screen'] = pred_screen
        output['pred_verts'] = pred_verts
        output['pred_tex'] = pred_tex
        output['mask'] = mask
        output['psnr'] = psnr
        output['ssim'] = ssim
        output['psnr_mask'] = psnr_m
        output['ssim_mask'] = ssim_m

        return losses, output

    def save_img(data, output, i, key, tag=''):
        screen_mask = data['screen_mask'][i].detach().cpu()
        gt_screen = data['photo'][i] * 255
        if output['pred_screen'][i] is not None:
            pred_screen = torch.clamp(output['pred_screen'][i] * 255, 0, 255)
            # save the tensor
            if args.save_tensor:
                tensor_path = os.path.join(args.tensor_path, f"tensor_{tag}.pt")
                torch.save(pred_screen.cpu(), tensor_path)
            # apply gamma correction
            save_pred_image = pred_screen.detach().cpu().numpy().astype(np.uint8) 
            save_pred_image = (255 * gammaCorrect(save_pred_image / 255.0)).astype(np.uint8)
            Image.fromarray(save_pred_image).save(os.path.join(args.result_path, 'pred_%s.png' % tag))
        # apply gamma correction
        save_gt_image = gt_screen.detach().cpu().numpy().astype(np.uint8)
        save_gt_image = (255 * gammaCorrect(save_gt_image / 255.0)).astype(np.uint8)
        Image.fromarray(save_gt_image).save(os.path.join(args.result_path, 'gt_%s.png' % tag))
        lpips_loss = lpips.LPIPS(net='alex').cuda()
        lpips_score = lpips_loss(torch.from_numpy(save_pred_image).permute(2, 0, 1).unsqueeze(0).cuda(), torch.from_numpy(save_gt_image).permute(2, 0, 1).unsqueeze(0).cuda()).mean().item()

        diff = pred_screen.detach().cpu().numpy().astype(np.int16) - (gt_screen.detach().cpu().numpy() * screen_mask.detach().cpu().numpy()).astype(np.int16)
        # per-pixel L1 error
        err = np.mean(np.abs(diff), axis=2)  # H×W
        err_norm = cv2.normalize(err, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(err_norm, cv2.COLORMAP_JET)
        if i == 0:
            Image.fromarray(heatmap).save(os.path.join(args.result_path, f'errmap_{tag}.png'))

        max_val = np.max(np.abs(diff)) + 1e-6
        diff_norm = diff / max_val  
        diff_vis = ((diff_norm + 1) * 127.5).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(save_gt_image, 0.6, heatmap, 0.4, 0)
        if i == 0:
            Image.fromarray(overlay).save(os.path.join(args.result_path, f'overlay_{tag}.png'))

            plt.imshow(diff, cmap='seismic', vmin=-max_val, vmax=max_val)
            plt.colorbar(label='signed pixel error')
            plt.axis('off')
            plt.savefig(os.path.join(args.result_path, f'matdiff_{tag}.png'), bbox_inches='tight')
            plt.close()
        if key not in pred_imgs:
            pred_imgs[key] = []
        if key not in gt_imgs:
            gt_imgs[key] = []
        if key not in diff_imgs:
            diff_imgs[key] = []

        pred_imgs[key].append(os.path.join(args.result_path, 'pred_%s.png' % tag))
        gt_imgs[key].append(os.path.join(args.result_path, 'gt_%s.png' % tag))
        diff_imgs[key].append(os.path.join(args.result_path, 'diff_%s.png' % tag))
        return lpips_score

    def save_video(key):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("mp4v")
        fps = 12
        videowrite_path = os.path.join(args.video_path, "%s.mp4" % (key))
        
        cur_gt_img = gt_imgs[key].copy()
        cur_pred_img = pred_imgs[key].copy()
        cur_diff_img = diff_imgs[key].copy()
        cur_gt_img = sorted(cur_gt_img)
        cur_pred_img = sorted(cur_pred_img)
        cur_diff_img = sorted(cur_diff_img)

        assert(len(cur_gt_img) == len(cur_pred_img))
        assert(len(cur_diff_img) == len(cur_pred_img))
        for idx, img in enumerate(cur_gt_img):
            gt = cv2.imread(img)
            pred = cv2.imread(cur_pred_img[idx])

            if "GT" in args.sample_method:

                full_img = gt
            else:
                full_img = pred
            h, w = full_img.shape[:2]
            if idx == 0:
                videowrite = cv2.VideoWriter(videowrite_path, fourcc, fps, (w, h))
            videowrite.write(full_img)

        videowrite.release()
    

    val_idx = 0
    best_screen_loss = 1e8
    best_tex_loss = 1e8
    best_vert_loss = 1e8
    model.train()

    model.eval()
    iter = 1
    begin_time = time.time()

    gt_imgs = {}
    pred_imgs = {}
    diff_imgs = {}
    test_segment_full_name = set()

    # track the frame-wise error of the rendered output
    global prev
    global psnr_all
    global ssim_all
    global pmask_all
    global smask_all
    global psnr_img_m_all
    global ssim_img_m_all
    global psnr_img_all
    global ssim_img_all
    global lpips_all
    prev = None
    psnr_all = []
    ssim_all = []
    pmask_all = []
    smask_all = []
    psnr_img_m_all = []
    ssim_img_m_all = []
    psnr_img_all = []
    ssim_img_all = []
    lpips_all = []
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    pmask_avg = AverageMeter()
    smask_avg = AverageMeter()
    lpips_avg = AverageMeter()
    for j in range(iter):
        total, vert, tex, screen, kl = [], [], [], [], []
        for i, data in enumerate(visual_loader):
            losses, output = run_net(data)
            total.append(losses['total_loss'].item())
            vert.append(losses['vert_loss'].item())
            tex.append(losses['tex_loss'].item())
            screen.append(losses['screen_loss'].item())

            kl.append(losses['kl'].item())

            if args.report_psnr:
                pred_tex = output["pred_tex"]
                gt_tex = output["gt_tex"]
                mask = output['mask']

                
                # permute to (c, b, h, w)
                pred_tex = pred_tex.permute(1, 0, 2, 3)
                gt_tex = gt_tex.permute(1, 0, 2, 3)
                mask = mask.permute(1, 0, 2, 3)


                psnr, ssim, _ = eval_errors(pred_tex, gt_tex)
                
                pred_tex = pred_tex * mask
                gt_tex = gt_tex * mask

                psnr_mask, ssim_mask, _ = eval_errors(pred_tex, gt_tex) 
                # meter
                psnr_avg.update(psnr.item())
                ssim_avg.update(ssim.item())
                pmask_avg.update(psnr_mask.item())
                smask_avg.update(ssim_mask.item())

                # record
                psnr_all.append(psnr.item())
                ssim_all.append(ssim.item())
                pmask_all.append(psnr_mask.item())
                smask_all.append(ssim_mask.item())
                psnr_img_all.append(output['psnr'].item())
                ssim_img_all.append(output['ssim'].item())
                psnr_img_m_all.append(output['psnr_mask'].item())
                ssim_img_m_all.append(output['ssim_mask'].item())


            if i == args.val_num and j != (iter - 1):
                break
            if j == (iter - 1) and local_rank == 0:
                lp = 0
                for k in range(args.val_batch_size):
                    if str(data['exp'][k]) not in test_segment_full_name:
                        test_segment_full_name.add(str(data['exp'][k]))
                    lp_img = save_img(data, output, k, "%s_%s" % (str(data['exp'][k]), str(data['cam_idx'][k])),'val_%s_%s_%s' % (str(data['exp'][k]), str(data['cam_idx'][k]), str(data['frame'][k])))
                    lp+= lp_img
                lpips_all.append(lp/args.val_batch_size)
                if i > 1:
                    break
    for cam in [args.view_id]:
        for exp in tqdm(list(test_segment_full_name)):
            key = str(exp) + "_" + str(cam)
            save_video(key)

    if args.report_psnr:
        print(len(psnr_all),len(lpips_all))
        metrics = {
            "psnr": psnr_all,
            "ssim": ssim_all,
            "psnr_mask": pmask_all,
            "ssim_mask": smask_all,
            "lpips": lpips_all,
            "psnr_img": psnr_img_all,
            "ssim_img": ssim_img_all,
            "psnr_img_m": psnr_img_m_all,
            "ssim_img_m": ssim_img_m_all
        }

        filename = os.path.join(args.err_path, args.err_file)
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(filename)

    tex_loss = np.array(tex).mean()
    vert_loss = np.array(vert).mean()
    screen_loss = np.array(screen).mean()
    kl = np.array(kl).mean()

    val_idx += 1
    print('val %d vert %.3f tex %.3f screen %.5f kl %.3f' %
        (val_idx, vert_loss, tex_loss, screen_loss, kl))

    best_screen_loss = min(best_screen_loss, screen_loss)
    best_tex_loss = min(best_tex_loss, tex_loss)
    best_vert_loss = min(best_vert_loss, vert_loss)

    end_time = time.time()
    print('Testing takes %f seconds' % (end_time - begin_time))
    print('best screen loss %f, best tex loss %f best vert loss %f \n' % (best_screen_loss, best_tex_loss, best_vert_loss))

    if args.report_psnr:
        print(f"PSNR = {psnr_avg.avg} | SSIM = {ssim_avg.avg} | Masked = {pmask_avg.avg}, {smask_avg.avg}")


    return best_screen_loss, best_tex_loss, best_vert_loss


if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed run')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Validation batch size')
    parser.add_argument('--arch', type=str, default='base', help='Model architecture - base|warp|res|non|bilinear')
    parser.add_argument('--nlatent', type=int, default=256, help='Latent code dimension - 128|256')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--resolution', default=[2048, 1334], nargs=2, type=int, help='Rendering resolution')
    parser.add_argument('--tex_size', type=int, default=1024, help='Texture resolution')
    parser.add_argument('--mesh_inp_size', type=int, default=21918, help='Input mesh dimension')
    parser.add_argument('--data_dir', type=str, default='/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS', help='Directory to dataset root')
    parser.add_argument('--krt_dir', type=str, default='/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/KRT', help='Directory to KRT file')
    parser.add_argument('--loss_weight_mask', type=str, default='./loss_weight_mask.png', help='Mask for weighted loss of face')
    parser.add_argument('--framelist_test', type=str, default='/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt', help='Frame list for testing')
    parser.add_argument('--test_segment_config', type=str, default='/mnt/captures/ecwuu/test_segment.json', help='Directory of expression segments for visualization')
    parser.add_argument('--lambda_verts', type=float, default=1, help='Multiplier of vertex loss')
    parser.add_argument('--lambda_screen', type=float, default=1, help='Multiplier of screen loss')
    parser.add_argument('--lambda_tex', type=float, default=1, help='Multiplier of texture loss')
    parser.add_argument('--lambda_kl', type=float, default=1e-2, help='Multiplier of KL divergence')
    parser.add_argument('--max_iter', type=int, default=200000, help='Maximum number of training iterations, overrides epoch')
    parser.add_argument('--log_every', type=int, default=1000, help='Interval of printing training loss')
    parser.add_argument('--val_every', type=int, default=5000, help='Interval of validating on test set')
    parser.add_argument('--val_num', type=int, default=500, help='Number of iterations for validation')
    parser.add_argument('--n_worker', type=int, default=8, help='Number of workers loading dataset')
    parser.add_argument('--pass_thres', type=int, default=50, help='If loss is x times higher than the previous batch, discard this batch')
    parser.add_argument('--result_path', type=str, default='./runs/experiment', help='Directory to output files')
    parser.add_argument('--video_path', type=str, default='./runs/experiment', help='Directory to output videos')
    parser.add_argument('--tensor_path', type=str, default='./runs/experiment', help='Directory to output tensors')
    parser.add_argument('--err_path', type=str, default='./runs/experiment', help='Directory to errors')
    parser.add_argument('--err_file', type=str, default='./runs/experiment', help='filename of errors')
    parser.add_argument('--camera_config', type=str, default=None, help='Directory to camera set config file')
    parser.add_argument('--camera_setting', type=str, default=None, help='Key of camera setting to camera config file')
    parser.add_argument('--model_path', type=str, default=None, help='Model path')
    parser.add_argument('--sample_method', type=str, default=None, help='PTQ method')
    parser.add_argument('--calib_method', type=str, default=None, help='PTQ method')
    parser.add_argument('--save_video', type=bool, default=True, help='Save visualization as .mp4')
    parser.add_argument('--report_psnr', type=str2bool, nargs='?', const=True, default=False, help='Repoort PSNR')
    parser.add_argument('--save_tensor', type=str2bool, nargs='?', const=True, default=False, help='Save RGB tensor')
    parser.add_argument('--groupsize', type=int, default=-1, help='group size for activation quantization')
    parser.add_argument(
        '--sym', type=str2bool, default=False,
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--clip_ratio', type=float, default=1.0,
        help='Clip ratio for quantization.'
    )
    parser.add_argument(
        "--wbit", type=int, default=32, help="weight precision"
    )
    parser.add_argument(
        "--abit", type=int, default=32, help="activation precision"
    )
    parser.add_argument(
        "--view_id", type=str, default="400029", help="visualize view id"
    )
    parser.add_argument(
        '--colmn', type=str2bool, default=False,
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--use_2d_bound', type=str2bool, default=False,
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--row', type=str2bool, default=True,
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--omni', type=str2bool, default=False,
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    experiment_args = parser.parse_args()
    print(experiment_args)
    assert(experiment_args.camera_config != None)
    assert(experiment_args.test_segment_config != None)

    if experiment_args.camera_config is not None:
        f = open(experiment_args.camera_config, 'r')
        camera_config = json.load(f)
        f.close()
        if experiment_args.camera_setting is not None:
            camera_set = camera_config[experiment_args.camera_setting]
        else:
            camera_set = None
    else:
        camera_set = None

    f = open(experiment_args.test_segment_config, 'r')
    test_segment_config = json.load(f)
    f.close()
    test_segment = test_segment_config["segment"]


    main(experiment_args, camera_set, test_segment)
    print("visualization completed")
