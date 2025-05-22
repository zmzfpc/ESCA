import os 
import torch as th
import torch.nn.functional as thf
from tqdm import tqdm
from typing import Union, List
from qlib.base import QConvTranspose2dWN, QLinearWN
from qlib.ptq import ActQuant
from qlib.utils import DataSaverHook, AverageMeter
from qlib.qwrap import model2ptq, _parent_name
from qlib.utils import plot_masks
from qlib.gptq import *
from qlib.quant import *
import cv2
import nvdiffrast.torch as dr

def hessian_trace(layer, data, n_samples=32):
    trace = 0.0
    for i in range(n_samples):
        out = layer(data.requires_grad_())
        g   = torch.autograd.grad(out.sum(), data, retain_graph=True)[0]
        trace += (g ** 2).sum().item()  
    return trace / n_samples

class UVFetcher:
    """
    Modified on top of the default Renderer
    """
    def __init__(self):
        self.glctx = dr.RasterizeCudaContext()

    def render(self, M, pos, pos_idx, uv, uv_idx, tex, resolution=[2048, 1334]):
        ones = th.ones((pos.shape[0], pos.shape[1], 1)).to(pos.device)
        pos_homo = th.cat((pos, ones), -1)
        projected = th.bmm(M, pos_homo.permute(0, 2, 1))
        projected = projected.permute(0, 2, 1)
        proj = th.zeros_like(projected)
        proj[..., 0] = (
            projected[..., 0] / (resolution[1] / 2) - projected[..., 2]
        ) / projected[..., 2]
        proj[..., 1] = (
            projected[..., 1] / (resolution[0] / 2) - projected[..., 2]
        ) / projected[..., 2]
        clip_space, _ = th.max(projected[..., 2], 1, keepdim=True)
        proj[..., 2] = projected[..., 2] / clip_space

        pos_view = th.cat(
            (proj, th.ones(proj.shape[0], proj.shape[1], 1).to(proj.device)), -1
        )
        pos_idx_flat = pos_idx.view((-1, 3)).contiguous()
        uv_idx = uv_idx.view((-1, 3)).contiguous()
        # tex = tex.permute((0, 2, 3, 1)).contiguous()

        rast_out, rast_out_db = dr.rasterize(
            self.glctx, pos_view, pos_idx_flat, resolution
        )
        texc, _ = dr.interpolate(uv, rast_out, uv_idx)
        return texc


class PTQTrainer(object):
    """
    Trainer for post training quantization
    """
    def __init__(self, model:th.nn.Module, max_iter:int=100, dataloader=None, args=None, 
                texmean=None, texstd=None, vertmean=None, vertstd=None, logger=None) -> None:
        self.model = model
        self.logger = logger

        # args
        self.args = args

        self.use_2d_bound = args.use_2d_bound
        # post training quantization parameters
        self.percdamp = args.percdamp
        self.agroupsize = args.agroupsize
        self.wgroupsize = args.wgroupsize
        self.act_order = args.act_order
        self.static_groups = args.static_groups
        self.cr = args.clip_ratio
        # precision
        self.wbit = args.wbit
        self.abit = args.abit
        self.reversed = args.reversed
        self.omni = args.omni
        self.sym = args.sym
        self.perchannel = args.perchannel
        self.mse = args.mse
        
        self.new_arch = args.new_arch
        self.colmn = args.colmn
        self.row = args.row
        self.mask = args.mask
        if self.mask:
            self.kernel_size = 21
            self.sigma = 5.0
        
        self.train = args.train
        
        self.trits = args.trits
        # dataloader for clibration
        self.dataloader = dataloader



        # max epochs
        self.max_iter = max_iter

        # tau
        self.tau = args.tau
        # self.tau = 0.9
        # visibilitiy map
        self.uvfetcher = UVFetcher()
        self.frame_size = 1024

        # for renderer
        self.texmean = texmean
        self.texstd = texstd
        self.vertmean = vertmean
        self.vertstd = vertstd

        self.mask_weighted = args.mask_weighted
        # for mask visualization
        self.mask_path = os.path.join(self.args.result_path, "mask_1024x1024_")
        loss_weight_mask = cv2.flip(cv2.imread("./loss_weight_mask.png"), 0)
        loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
        loss_weight_mask = th.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float()
        self.loss_weight_mask = loss_weight_mask.mean(dim=1, keepdim=True)

    def forward_model(self, data):
        M = data["M"].cuda()
        vert_ids = data["vert_ids"].cuda()
        uvs = data["uvs"].cuda()
        uv_ids = data["uv_ids"].cuda()
        avg_tex = data["avg_tex"].cuda()
        mask = data["mask"].cuda()
        mask, _ = torch.max(mask, dim=1, keepdim=True)
        
        view = data["view"].cuda()
        verts = data["aligned_verts"].cuda()
        cams = data["cam"].cuda()
        output = {}

        if self.args.arch == "warp":
            pred_tex, pred_verts, unwarped_tex, warp_field, kl = self.model(
                avg_tex, verts, view, cams=cams
            )
            output["unwarped_tex"] = unwarped_tex
            output["warp_field"] = warp_field
        else:
            pred_tex, pred_verts, kl = self.model(avg_tex, verts, view, cams=cams)

        pred_verts = pred_verts * self.vertstd + self.vertmean
        pred_tex = (pred_tex * self.texstd + self.texmean) / 255.


        if self.mask:
        # renderer
            uv_mask = self.uvfetcher.render(M, pred_verts, vert_ids, uvs, uv_ids, pred_tex, self.args.resolution)
            output["uv"] = uv_mask
            output["texture_mask"] = mask

        
        output["pred_verts"] = pred_verts
        output["pred_tex"] = pred_tex
        
        
        # for decoder model reconstruction
        output["view"] = view

        th.cuda.empty_cache()

        return output

    def fetch_layer_data(self, layer: Union[QConvTranspose2dWN, QLinearWN], batch, model_calib=False):
        hook = DataSaverHook(store_input=True, store_output=True)
        handle = layer.register_forward_hook(hook)

        with th.no_grad():
            output = self.forward_model(batch)

        handle.remove()
        if self.mask:
            if not model_calib:
                return hook.input[0], hook.output, output["uv"].detach(), output["texture_mask"].detach()
            else:
                return hook.input[0], output["uv"].detach(), hook.output, output["uv"].detach()
        else:
            if not model_calib:
                return hook.input[0], hook.output, None, None 
            else:
                return hook.input[0], None, hook.output, None
        
    def uv2idx(self, uv:th.Tensor):
        """
        Conver the uv grid into the indexes

        Args:
        - uv (Tensor): vt_img that represents the visibility map 

        Output:
        - Indexes that generated from the normalized visibility map (grid)
        """

        idx = uv.mul(self.frame_size-1).round().clamp_max(self.frame_size-1)
        return idx.int()

    def mask_indexing(self, uv:th.Tensor, weighted=False):
        """
        Highlighting the pixels based on indexes
        """
        mask = th.zeros(self.frame_size, self.frame_size).cuda()

        uvf = uv.view(-1, 2)
        uvx = uvf[:, 0].long()
        uvy = uvf[:, 1].long()
        
        if weighted:
            mask.index_put_(
            indices=(uvy, uvx),
            values=torch.ones_like(uvy, dtype=torch.float32),
            accumulate=True 
            )
            mask[0][0] = 1

            mask = mask / mask.max() 
        else:
            mask[uvy, uvx] = 1.0

        return mask
    
    def uv2masks(self, ind:th.Tensor):
        masks = []
        for idx in ind:
            mask = self.mask_indexing(idx, weighted=self.mask_weighted)
            masks.append(mask.unsqueeze(0))
        
        masks = th.cat(masks, dim=0)
        return masks
    
    def fetch_layer_data_all(self, layer: Union[QConvTranspose2dWN, QLinearWN], layer_name):
        cached_batches = []
        self.uv_masks = []
        self.texture_masks = []
        for i, batch in enumerate(tqdm(self.dataloader)):
            x, y, uv, texture_uv = self.fetch_layer_data(layer, batch)

            if self.mask:
                ind = self.uv2idx(uv)
                masks = self.uv2masks(ind)
                self.uv_masks.append(masks.unsqueeze(1))

                self.texture_masks.append(texture_uv)

            cached_batches.append((x, y))
        
            
        self.logger.info(f"Data Fetched for layer {layer_name}!")
        
        th.cuda.empty_cache()
        return cached_batches


        self.logger.info(f"Data Fetched for layer {layer_name}! | last batch of mask: {self.mask_path + layer_name + 'uv.png'}")
        th.cuda.empty_cache()
        return cached_batches

    
    def downsample_mask(self, uv:th.Tensor, height):
        # stride and window
        kernel = uv.size(2) // height
        mask = thf.max_pool2d(uv, kernel_size=kernel, stride=kernel)
        return mask.cuda()
    
    def shape_wise_filter(self, y:th.Tensor, tau=None):
        """
        Pixel filter for calibration:
        Filter out the 1x1xC pixels that has the long-tailed distribution

        The shape-wise std scores are considered as the metric for filtering
        The sparsity of the masking is controlled by tau. 
        """
        assert len(y.size()) == 4, "The shape of the feature map tensor must be 4-D"
        if tau is None:
            tau = self.tau
        num_pixel_keep = int(y.size(2)*y.size(3) * tau)

        b, c, h, w = y.size()
        yd = y.detach()

        with th.no_grad():
            masks = []

            # row channel plane
            ystd = yd.std(dim=[1])

            for i in range(b):
                mask = th.zeros(h, w)
                scores = ystd[i]

                # top-k score
                tpk_score = th.topk(scores.view(-1), num_pixel_keep, sorted=True)
                threshold = tpk_score.values[-1]

                # mask[scores.lt(threshold), :] = 1.0
                mask = scores.lt(threshold).float()
                masks.append(mask.unsqueeze(0).unsqueeze(0))
        
        return th.cat(masks, dim=0).cuda()
    
    def row_wise_filter(self, y:th.Tensor):
        """
        Pixel filter for calibration:
        Filter out the row-channel pixels that has the long-tailed distribution

        The std scores of the row-channel plane are considered as the metric for filtering
        The sparsity of the masking is controlled by tau. 

        Args:
        - tau: float, tunnable parameter that controls the intensity of filtering

        Output:
        - Feature map masks with the long-tailed distribution filtered. 
        """
        assert len(y.size()) == 4, "The shape of the feature map tensor must be 4-D"

        num_rows_keep = int(y.size(3) * self.tau)

        b, c, h, w = y.size()
        yd = y.detach()

        with th.no_grad():
            masks = []

            # row channel plane
            ystd = yd.std(dim=[1,3])

            for i in range(b):
                mask = th.zeros(h, w)
                scores = ystd[i]

                # top-k score
                tpk_score = th.topk(scores, num_rows_keep, sorted=True)
                threshold = tpk_score.values[-1]

                # activate rows
                mask[scores.lt(threshold), :] = 1.0
                masks.append(mask.unsqueeze(0).unsqueeze(0))
        
        return th.cat(masks, dim=0).cuda()

    
    def layer_reconstruction(self, layer:Union[QConvTranspose2dWN, QLinearWN], layer_name, lr, cached_data:List):
        # freeze the weights and bias
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
        # layer.g.requires_grad = False



        if self.abit < 32:
            layer.xq = ActQuant(nbit=self.abit, gp=self.agroupsize, sym= self.sym, cr= self.cr).cuda()
        else:
            print("FULL")
        
        if self.train:
            qparam = []
            qparam.append(layer.x_smooth)
            optimizer = th.optim.Adam(qparam, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        
            # scheduler
            scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter, eta_min=0.)

            # # loss function
            loss_fn = th.nn.MSELoss()

            if isinstance(layer, (QConvTranspose2dWN, QLinearWN)):
                loss = AverageMeter()
                spars = AverageMeter()
                pbar = tqdm(range(self.max_iter))
                for e in pbar:
                    for idx, batch in enumerate(cached_data):
                        x, y = batch 

                    # cuda 
                        x = x.cuda()
                        y = y.cuda()

                    # forward pass 
                        out = layer(x, ptq_training=True)

                        # visibility masks
                        s = 0
                        if isinstance(layer, QConvTranspose2dWN):
                            if self.tau > 0:
                                visib = self.uv_masks[idx]
                                loss_weight_mask = self.loss_weight_mask.squeeze(0).cuda()

                                ymask = self.downsample_mask(visib, height=y.size(2))

                            # std filtering
                                filter_mask = self.shape_wise_filter(y)
                                ymask = ymask.mul(filter_mask)

                            # sparsity
                                s = ymask[ymask.eq(0.)].numel() / ymask.numel()
                                spars.update(s)

                                y = y.mul(ymask)
                                out = out.mul(ymask)
                            else:
                                s = 0.0
                        else:
                            # no sparsity on linear layer
                            s = 0.0
                        rec_loss = loss_fn(out, y)
                        loss.update(rec_loss.item())
                        optimizer.zero_grad()
                        # rec_loss.backward(retain_graph=True)
                        rec_loss.backward()
                        optimizer.step()

                

                    if e % 50 == 0:
                        print("Rec loss:", loss.avg)

                    scheduler.step()
                    pbar.set_description(f"Rec loss:{layer_name} = {loss.avg:.4e} | yspars = {s:.3f} | tau = {self.tau}")
  

        self.logger.info(f"{layer_name} Done!")

        th.cuda.empty_cache()

        return layer

    def nnfreeze(self, module:th.nn.Module):
        """
        Freeze the vanilla model before assigning the quantizers
        """
        for n, p in module.named_parameters():
            if p.requires_grad:
                p.requires_grad = False
        




    def fit(self):
        # convert the vanilla modules to quantization-ready modules
        tex_dec = self.model.module.dec
        
        # freeze all the weights and bias from learning
        self.nnfreeze(tex_dec)



        qtex_dec = model2ptq(tex_dec, qbit=self.abit)

        # insert the low precision decoder back
        self.model.module.dec = qtex_dec
        modules = dict(qtex_dec.named_modules(remove_duplicate=False))
        # self.model.module.dec = qtex_dec

        # map to cuda
        self.model = self.model.cuda()

        if self.reversed:
            layers = reversed(list(modules.items()))
        else:
            layers = modules.items()

        self.model = self.model.cuda()
        # for n, m in layers:
        #     if isinstance(m, (Union[DeconvTexelBias])):
        #         cached_data = self.fetch_layer_data_all(m, n)
        #         for idx, (x, y) in enumerate(cached_data):
        #             x_stats = analyze_and_save_distribution(x, path =self.mask_path + n +str(idx) +"_DeconvTexelBiasX.pdf",clip_ratio=0) 
        #             y_stats = analyze_and_save_distribution(y, path =self.mask_path + n +str(idx) +"_DeconvTexelBiasY.pdf",clip_ratio=0)
        #             plot_activation_outliers_per_channel(x, path=self.mask_path + n +str(idx) +"_DeconvTexelBiasP.pdf")

        for n, m in layers:
            if isinstance(m, (Union[QConvTranspose2dWN])):

                if not "texture_fc" in n:
                    print(f"Layer {n} is being calibrated!")

                    cached_data = self.fetch_layer_data_all(m, n)
                    if self.omni: 
                        loss_weight_mask = self.loss_weight_mask.squeeze(0).cuda()

                        roi = self.downsample_mask(loss_weight_mask, height=cached_data[0][0].size(2)).unsqueeze(0)

                        plot_masks(roi.squeeze(1), self.mask_path + n +"_mask.png")
                        m.set_smooth(alpha=0.8, cached_data=cached_data, weight=roi, k_percent=self.tau*10)


                    gptq = GPTQ(m) 
                    gptq.quantizer = Quantizer()
                    gptq.quantizer.configure(
                        self.wbit, perchannel=self.perchannel, sym=self.sym, mse=self.mse, trits=self.trits
                    )
                    for idx, (x, y) in enumerate(cached_data):
                        
   
                        if self.omni:
                            x = x.cuda() / m.x_smooth.view(1,-1, 1, 1)

                        else:
                            x = x.cuda()
                            
                        y = y.cuda()

                        if self.mask:
                            visib = self.uv_masks[idx]
                            
                            
                            loss_weight_mask = self.loss_weight_mask.squeeze(0).cuda()
                            face_mask = visib.mul(loss_weight_mask)
                            if self.new_arch:
                                visib = face_mask
                
                        else:
                            visib = None

                        gptq.add_batch(x, y, visib, None, tau=1 - self.cr)                        
                    if self.wbit < 32:
                        new_layer = gptq.fasterquant(
                            percdamp=self.percdamp, groupsize=self.wgroupsize, actorder=self.act_order, static_groups=self.static_groups
                    )
                    else:
                        print("FULL")
                        new_layer = m
                    new_layer = self.layer_reconstruction(new_layer, n, lr=self.args.lr, cached_data=cached_data)

                    parent_name, name = _parent_name(n)

                    setattr(modules[parent_name], name, new_layer)

                gptq.free()
                del gptq


                del cached_data
        
        torch.cuda.empty_cache()
        
        return self.model
 