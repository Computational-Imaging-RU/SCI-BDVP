from cacti.utils.utils import *
import torch 
import einops
import cv2 
import numpy as np 
from cacti.models.dip import autoencodernet,UNet_noskip
from cacti.models.tv import TV
import logging
from cacti.utils.metrics import ssim_torch,ssim_block
import torchvision

def model_load(B,lrnum,channel,kernel,device):
    # model = UNet_noskip(B, B, bilinear=False).to(device)
    up = channel
    model = autoencodernet(B, up, kernel_size=kernel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lrnum, betas=(0.9, 0.999))
    loss_fn = torch.nn.MSELoss().to(device)
    return model, optimizer, loss_fn


def get_noise(data_size, channal,device, type='u', var=1./10):
    # shape = [data_size[0], data_size[1], data_size[2], data_size[3]]
    upsample_scale = 2**(len(channal)-1)
    input_channel = channal[0]
    shape = [data_size[0], input_channel, data_size[2]//upsample_scale, data_size[3]//upsample_scale]
    net_input = torch.zeros(shape)
    if type == 'u':
        net_input = net_input.uniform_()*var
    elif type == 'n':
        net_input = net_input.normal_()*var
    else:
        assert False
    return net_input.to(device).float()

def nonlinear_tv_loss(video, tv_weight, eps=1e-3):
    batch,frame,h,w = video.shape
    diff_f = video[:, :-1, :, :] - video[:, 1:, :, :]  # Temporal gradient (F-1, H, W)
    # diff_h = video[:, :, :-1, :] - video[:, :, 1:, :]  # Height gradient (F, H-1, W)
    # diff_w = video[:, :, :, :-1] - video[:, :, :, 1:]  # Width gradient (F, H, W-1)

    # Nonlinear function of the gradients - this could be adapted for different behaviors
    # grad_mag_sq = diff_f.pow(2).sum(dim=1, keepdim=True) + 0.1*diff_h.pow(2).sum(dim=2, keepdim=True) + 0.1*diff_w.pow(2).sum(dim=3, keepdim=True)
    grad_mag_sq = diff_f.pow(2).sum(dim=1, keepdim=True)
    penalty = torch.sqrt(grad_mag_sq + eps)

    total_variation = tv_weight * torch.sum(penalty)
    return total_variation / (batch*frame*h*w)

def DIP_denoiser(truth_tensor, net_input, ref_truth, Phi_tensor, y_tensor, model, optimizer, loss_fn, iter_num, y_w, tv_w,device):

    logger = logging.getLogger()
    loss_min = torch.tensor([100]).to(device).float()
    for i in range(iter_num):

        model_out = model(net_input)
        optimizer.zero_grad()
        x_loss = loss_fn(model_out, ref_truth) 
        
        # model_out_y = model_out[:,0::4,:,:]
        y_hat = A_torch(model_out, Phi_tensor)
        # y_hat = A_torch(model_out_y, Phi_tensor)
        y_loss = loss_fn(y_hat, y_tensor) 
        tv_loss = nonlinear_tv_loss(model_out, 0.05)
        loss = x_loss + y_w*y_loss + tv_w*tv_loss
        groud_truth_loss = loss_fn(model_out, truth_tensor.unsqueeze(0))
        loss.backward()
        optimizer.step()

        if (i+1)%25==0 and y_loss < loss_min*1.1:
        # if (i+1)%25==0:
            #loss_min = loss
            loss_min = y_loss
            output = model_out.detach().cpu().numpy()
        if (i+1)%50==0:
            out_PSNR = psnr_torch(truth_tensor, torch.squeeze(model_out))
            out_SSIM = ssim_torch(truth_tensor, torch.squeeze(model_out))
            logger.info('DIP iter {}, x_loss = {:.5f}, y_loss = {:.5f}, tv_loss = {:.5f}, groud_truth_loss = {:.5f}, PSNR = {:.4f}, SSIM = {:.4f}'.format(i+1, x_loss.detach().cpu().numpy(), y_loss.detach().cpu().numpy(), tv_loss.detach().cpu().numpy(), groud_truth_loss.detach().cpu().numpy(), out_PSNR.detach().cpu().numpy(), out_SSIM))
    return output, loss_min.detach().cpu().numpy()

def find_optimal_lambda(x, g, y, mask, lambdas = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])):

    min_value = float('inf')
    optimal_lambda = None
    x_new = None

    for lambda1 in lambdas:
        temp_x = x + lambda1 * g
        objective_value = (y - A(temp_x,mask)).norm()  # Calculate the norm of the difference
        print(lambda1,objective_value)
        if objective_value < min_value:
            min_value = objective_value
            optimal_lambda = lambda1
            x_new = temp_x

    return optimal_lambda.item(), x_new

def augment_frames(data):
    rotations = [0, 90, 180, 270]
    transformed_frames = []

    for frame in data.squeeze(0):
        # print(frame.shape)
        pil_frame = torchvision.transforms.functional.to_pil_image(frame)
        for angle in rotations:
            # Apply rotation
            rotated_frame = torchvision.transforms.functional.rotate(pil_frame, angle)
            rotated_frame = torchvision.transforms.functional.to_tensor(rotated_frame).squeeze(0)
            # print(rotated_frame.shape)
            transformed_frames.append(rotated_frame)
            # print(transformed_frames[0].shape)

    # Stack all transformed frames along a new dimension
    augmented_data = torch.stack(transformed_frames, dim=0).unsqueeze(0)  # Add back the batch dimension
    return augmented_data


def padding(input_tensor, padding_size):
    # truth_tensor & y_tensor
    if len(input_tensor.size()) == 3:
        input_dim_1 = input_tensor.size(dim=0)
        input_dim_2 = input_tensor.size(dim=1)
        input_dim_3 = input_tensor.size(dim=2)

        # padding top & bottom & left & right bars
        padding_top = input_tensor[:, padding_size - 1, :].resize(input_dim_1, 1, input_dim_3)
        padding_bottom = input_tensor[:, input_dim_2 - 1, :].resize(input_dim_1, 1, input_dim_3)
        padding_left = input_tensor[:, :, padding_size - 1].resize(input_dim_1, input_dim_2, 1)
        padding_right = input_tensor[:, :, input_dim_3 - 1].resize(input_dim_1, input_dim_2, 1)
        for i in range(padding_size - 1):
            padding_top = torch.cat((padding_top, input_tensor[:, padding_size - i - 2, :].resize(input_dim_1, 1, input_dim_3)), 1)
            padding_bottom = torch.cat((padding_bottom, input_tensor[:, input_dim_2 - i - 2, :].resize(input_dim_1, 1, input_dim_3)), 1)
            padding_left = torch.cat((padding_left, input_tensor[:, :, padding_size - i - 2].resize(input_dim_1, input_dim_2, 1)), 2)
            padding_right = torch.cat((padding_right, input_tensor[:, :, input_dim_3 - i - 2].resize(input_dim_1, input_dim_2, 1)), 2)

        # padding four corners
        padding_top_left = torch.flip(input_tensor[:, : padding_size, : padding_size], [1, 2])
        padding_top_right = torch.flip(input_tensor[:, : padding_size, input_dim_3 - padding_size :], [1, 2])
        padding_bottom_left = torch.flip(input_tensor[:, input_dim_2 - padding_size :, : padding_size], [1, 2])
        padding_bottom_right = torch.flip(input_tensor[:, input_dim_2 - padding_size :, input_dim_3 - padding_size :], [1, 2])
        
        # concatenate
        output_tensor = torch.cat(
                                (torch.cat((padding_top_left, padding_top, padding_top_right), 2), 
                                 torch.cat((padding_left, input_tensor, padding_right), 2), 
                                 torch.cat((padding_bottom_left, padding_bottom, padding_bottom_right), 2)), 1)
        
    # ref_t & Phi_tensor
    elif len(input_tensor.size()) == 4:
        input_dim_1 = input_tensor.size(dim=0)
        input_dim_2 = input_tensor.size(dim=1)
        input_dim_3 = input_tensor.size(dim=2)
        input_dim_4 = input_tensor.size(dim=3)

        # padding top & bottom & left & right bars
        padding_top = input_tensor[:, :, padding_size - 1, :].resize(input_dim_1, input_dim_2, 1, input_dim_4)
        padding_bottom = input_tensor[:, :, input_dim_3 - 1, :].resize(input_dim_1, input_dim_2, 1, input_dim_4)
        padding_left = input_tensor[:, :, :, padding_size - 1].resize(input_dim_1, input_dim_2, input_dim_3, 1)
        padding_right = input_tensor[:, :, :, input_dim_4 - 1].resize(input_dim_1, input_dim_2, input_dim_3, 1)
        for i in range(padding_size - 1):
            padding_top = torch.cat((padding_top, input_tensor[:, :, padding_size - i - 2, :].resize(input_dim_1, input_dim_2, 1, input_dim_4)), 2)
            padding_bottom = torch.cat((padding_bottom, input_tensor[:, :, input_dim_3 - i - 2, :].resize(input_dim_1, input_dim_2, 1, input_dim_4)), 2)
            padding_left = torch.cat((padding_left, input_tensor[:, :, :, padding_size - i - 2].resize(input_dim_1, input_dim_2, input_dim_3, 1)), 3)
            padding_right = torch.cat((padding_right, input_tensor[:, :, :, input_dim_4 - i - 2].resize(input_dim_1, input_dim_2, input_dim_3, 1)), 3)

        # padding four corners
        padding_top_left = torch.flip(input_tensor[:, : , : padding_size, : padding_size], [2, 3])
        padding_top_right = torch.flip(input_tensor[:, : , : padding_size, input_dim_4 - padding_size :], [2, 3])
        padding_bottom_left = torch.flip(input_tensor[:, :, input_dim_3 - padding_size :, : padding_size], [2, 3])
        padding_bottom_right = torch.flip(input_tensor[:, :, input_dim_3 - padding_size :, input_dim_4 - padding_size :], [2, 3])
        
        # concatenate
        output_tensor = torch.cat(
                                (torch.cat((padding_top_left, padding_top, padding_top_right), 3), 
                                 torch.cat((padding_left, input_tensor, padding_right), 3), 
                                 torch.cat((padding_bottom_left, padding_bottom, padding_bottom_right), 3)), 2)

    return output_tensor


def GAP_denoise_dip(x,y,y1,mask,mask_s,dip_iter,x_ori, it_dip_outer, y_w, tv_w,lambda1,patch_list,lr,TV_pre_denoise,TV_iter,channel,kernel,aug,weight,noise_type,decrease,device,
                    multichannel=True,accelerate=True,color_denoiser=False,bayer=None,use_cv2_demosaic=True,gamma=0.01):
    
    logger = logging.getLogger()

    batch_size,frames,height,width = mask.shape
    
    truth_tensor = torch.from_numpy(np.transpose(x_ori, (2, 0, 1))).to(device).float()
    y_tensor = y.squeeze(0).to(torch.float32)
    Phi_tensor = mask

    src_x = torch.zeros_like(x).to(x.device)#torch.Size([1, 8, 256, 256])
    yb = A(x,mask)
    if accelerate: # accelerated version of GAP
        y1 = y1 + (y-yb)
        g = At((y1-yb)/mask_s,mask)
        # x  = x + lambda*g
        if it_dip_outer>200:
            lambda1,x = find_optimal_lambda(x, g, y, mask)
        else:
            x = x + lambda1*(At((y1-yb)/mask_s,mask)) # GAP_acc
    else:
        x = x + lambda1*(At((y-yb)/mask_s,mask)) # GAP
    
    

    #denoise
    
    if color_denoiser:
        assert bayer is not None,"Bayer is None"
        x_rgb = torch.zeros([frames,height*2, width*2,3]).to(mask.device)
        x_bayer = torch.zeros([frames,height*2, width*2]).to(mask.device)
        for ib in range(len(bayer)): 
            b = bayer[ib]
            x_bayer[:,b[0]::2, b[1]::2] = x[ib,:]
        for imask in range(frames):
            np_x_bayer = x_bayer[imask].cpu().numpy()
            if not use_cv2_demosaic:
                np_x_bayer = demosaicing_bayer(np_x_bayer)
            else:
                np_x_bayer = cv2.cvtColor(np.uint8(np.clip(np_x_bayer,0,1)*255), cv2.COLOR_BAYER_RG2BGR)
                np_x_bayer = np_x_bayer.astype(np.float32)
                np_x_bayer /= 255.
            x_rgb[imask] = torch.from_numpy(np_x_bayer).to(mask.device)
        x = einops.rearrange(x_rgb,"f h w c->1 f h w c")
    else:
        x = einops.rearrange(x,"b f h w->b f h w 1")#[1,8,256,256,1]

    # reference ground truth
    ref_t = einops.rearrange(x,"b f h w 1->b f h w")#[1,8,256,256]
    ref_t = ref_t.to(torch.float32)
    ref_t = torch.clamp(ref_t,0,1)
    ref_t_np = einops.rearrange(x,"1 f h w 1-> h w f").detach().cpu().numpy()
    print('psnr refernce truth',psnr_block(x_ori,ref_t_np), psnr_torch(truth_tensor, ref_t.squeeze(0)).detach().cpu().numpy())
    logger.info('Iteration {}, PSNR gradient = {:.4f}, SSIM gradient = {:.4f}, step_size = {:.1f}'.format(it_dip_outer, psnr_torch(truth_tensor, ref_t.squeeze(0)),ssim_torch(truth_tensor, ref_t.squeeze(0)), lambda1))

    if TV_pre_denoise:
        if it_dip_outer<TV_iter+1:
            tv_denoiser=TV(tv_weight=0.1, tv_iter_max=10)
            x=tv_denoiser(ref_t)
            logger.info('Iteration {}, PSNR reft+TV = {:.4f}, SSIM reft+TV = {:.4f}'.format(it_dip_outer, psnr_torch(truth_tensor, ref_t.squeeze(0)),ssim_torch(truth_tensor, ref_t.squeeze(0))))
            loss_func = torch.nn.MSELoss().to(device)
            loss_y_iter = loss_func(A_torch(x, Phi_tensor), y_tensor).detach().cpu().numpy()
        else:
            out_list = []
            loss_y_list = []
            for patch_size in patch_list:
                patch_num = height // patch_size
                out_list_per_batch_size = []
                loss_y_per_patch_size = 0
                padding_size = patch_size // 8
                for patch_w_id in range(patch_num):
                    for patch_h_id in range(patch_num):
                        logger.info('Patch size {}, Patch index [{}/{}]'.format(patch_size, patch_w_id * patch_num + patch_h_id + 1, patch_num * patch_num))
                        truth_tensor_padding = padding(truth_tensor, padding_size)
                        truth_tensor_patch = truth_tensor_padding[: , patch_w_id * patch_size : (patch_w_id + 1) * patch_size + 2 * padding_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size + 2 * padding_size]
                        ref_t_padding = padding(ref_t, padding_size)
                        ref_t_patch = ref_t_padding[:, :, patch_w_id * patch_size : (patch_w_id + 1) * patch_size + 2 * padding_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size + 2 * padding_size]
                        Phi_tensor_padding = padding(Phi_tensor, padding_size)
                        Phi_tensor_patch = Phi_tensor_padding[:, :, patch_w_id * patch_size : (patch_w_id + 1) * patch_size + 2 * padding_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size + 2 * padding_size]                        
                        y_tensor_padding = padding(y_tensor, padding_size)
                        y_tensor_patch = y_tensor_padding[:, patch_w_id * patch_size : (patch_w_id + 1) * patch_size + 2 * padding_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size + 2 * padding_size]
                        frames = ref_t.shape[1]
                        if patch_size==height:
                            model, optimizer, loss_fn = model_load(frames,lr,channel,kernel,device)
                            net_input_patch = get_noise(ref_t_patch.shape,channel,device,type=noise_type)                      
                            out_per_patch, loss_y_per_patch = DIP_denoiser(truth_tensor_patch, net_input_patch, ref_t_patch, Phi_tensor_patch, 
                                                                        y_tensor_patch, model,optimizer, loss_fn, 
                                                                        dip_iter*2,y_w, tv_w,device)
                        else:
                            model, optimizer, loss_fn = model_load(frames,lr,channel,kernel,device)
                            net_input_patch = get_noise(ref_t_patch.shape,channel,device,type=noise_type)                      
                            out_per_patch, loss_y_per_patch = DIP_denoiser(truth_tensor_patch, net_input_patch, ref_t_patch, Phi_tensor_patch, 
                                                                        y_tensor_patch, model,optimizer, loss_fn, 
                                                                        dip_iter,y_w, tv_w,device)
                        
                        out_list_per_batch_size.append(torch.from_numpy(out_per_patch[:, :, padding_size : - padding_size, padding_size : - padding_size]).to(x.device))
                        loss_y_per_patch_size += loss_y_per_patch
                        
                loss_y_list.append(loss_y_per_patch_size)
                out_inner_list_per_patch_size = []
                for patch_w_id in range(patch_num):
                    patch_w_start_id = patch_w_id * patch_num
                    out_inner = out_list_per_batch_size[patch_w_start_id] 
                    for patch_h_id in range(1, patch_num):
                        out_inner = torch.cat((out_inner, out_list_per_batch_size[patch_w_start_id + patch_h_id]), 3)
                    out_inner_list_per_patch_size.append(out_inner)
                out = out_inner_list_per_patch_size[0]
                for out_inner_id in range(1, len(out_inner_list_per_patch_size)):
                    out = torch.cat((out, out_inner_list_per_patch_size[out_inner_id]), 2)
                    # out = torch.clamp(out,0,1)
                # out = out[:, 0::4, :, :]
                out_list.append(out)
                psnr_per_patch_size = psnr_torch(truth_tensor, out.squeeze(0))
                ssim_per_patch_size = ssim_torch(truth_tensor, out.squeeze(0))
                logger.info('Patch size {}, y_loss_per_patch_size = {:.5f}, PSNR_per_patch_size = {:.4f}, SSIM_per_patch_size = {:.4f}'.format(patch_size, loss_y_per_patch_size, psnr_per_patch_size,ssim_per_patch_size))
            
            out = torch.zeros(1, 8, 256, 256).to(x.device)
            loss_y_iter = 0
            for patch_size_id in range(len(patch_list)):
                out += out_list[patch_size_id]
                loss_y_iter += loss_y_list[patch_size_id]
            out /= len(patch_list)
            loss_y_iter /= len(patch_list)

            logger.info('Iteration {}, PSNR DIP = {:.4f}, SSIM DIP = {:.4f}'.format(it_dip_outer, psnr_torch(truth_tensor, out.squeeze(0)),ssim_torch(truth_tensor, out.squeeze(0))))

            # x_rec = torch.from_numpy(x_rec).to(x.device)
            # print(x_rec.shape)
            
            x = einops.rearrange(out,"1 f h w->1 f h w 1",f=frames)
            # x = x.view(batch_size,frames,height,width)
            if color_denoiser:
                src_x[0] = x[0,:,0::2,0::2,0] 
                src_x[1] = x[0,:,0::2,1::2,1] 
                src_x[2] = x[0,:,1::2,0::2,1] 
                src_x[3] = x[0,:,1::2,1::2,2]
                x = src_x 
            else:
                x = einops.reduce(x,"b f h w c->b f h w","max")

    
    # if aug:
    #     ref_t_aug = augment_frames(ref_t).to(x.device)
        
    else:
        out_list = []
        loss_y_list = []
        for patch_size in patch_list:
            patch_num = height // patch_size
            out_list_per_batch_size = []
            loss_y_per_patch_size = 0

            padding_size = patch_size // 8

            # model, optimizer, loss_fn = model_load(frames)

            for patch_w_id in range(patch_num):
                for patch_h_id in range(patch_num):

                    # print(f"Patch size: {patch_size} - Patch index: [{patch_height_id * patch_num + patch_width_id}/{patch_num * patch_num}]")
                    logger.info('Patch size {}, Patch index [{}/{}]'.format(patch_size, patch_w_id * patch_num + patch_h_id + 1, patch_num * patch_num))

                    truth_tensor_padding = padding(truth_tensor, padding_size)
                    truth_tensor_patch = truth_tensor_padding[: , patch_w_id * patch_size : (patch_w_id + 1) * patch_size + 2 * padding_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size + 2 * padding_size]
                    # truth_tensor_patch = truth_tensor[:, patch_w_id * patch_size : (patch_w_id + 1) * patch_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size]
                    
                    # net_input_patch_size = patch_size // frames
                    # net_input_patch = net_input[:, :, patch_w_id * net_input_patch_size : (patch_w_id + 1) * net_input_patch_size, patch_h_id * net_input_patch_size : (patch_h_id + 1) * net_input_patch_size]

                    ref_t_padding = padding(ref_t, padding_size)
                    ref_t_patch = ref_t_padding[:, :, patch_w_id * patch_size : (patch_w_id + 1) * patch_size + 2 * padding_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size + 2 * padding_size]
                    # ref_t_patch = ref_t[:, :, patch_w_id * patch_size : (patch_w_id + 1) * patch_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size]

                    Phi_tensor_padding = padding(Phi_tensor, padding_size)
                    Phi_tensor_patch = Phi_tensor_padding[:, :, patch_w_id * patch_size : (patch_w_id + 1) * patch_size + 2 * padding_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size + 2 * padding_size]
                    # Phi_tensor_patch = Phi_tensor[:, :, patch_w_id * patch_size : (patch_w_id + 1) * patch_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size]
                    
                    y_tensor_padding = padding(y_tensor, padding_size)
                    y_tensor_patch = y_tensor_padding[:, patch_w_id * patch_size : (patch_w_id + 1) * patch_size + 2 * padding_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size + 2 * padding_size]
                    # y_tensor_patch = y_tensor[:, patch_w_id * patch_size : (patch_w_id + 1) * patch_size, patch_h_id * patch_size : (patch_h_id + 1) * patch_size]
                    frames = ref_t.shape[1]
                    if patch_size==height:

                        model, optimizer, loss_fn = model_load(frames,lr,channel,kernel,device)
                        net_input_patch = get_noise(ref_t_patch.shape,channel,device,type=noise_type)                      
                        out_per_patch, loss_y_per_patch = DIP_denoiser(truth_tensor_patch, net_input_patch, ref_t_patch, Phi_tensor_patch, 
                                                                    y_tensor_patch, model,optimizer, loss_fn, 
                                                                    dip_iter*2,y_w, tv_w,device)
                    else:
                        model, optimizer, loss_fn = model_load(frames,lr,channel,kernel,device)
                        net_input_patch = get_noise(ref_t_patch.shape,channel,device,type=noise_type)                      
                        out_per_patch, loss_y_per_patch = DIP_denoiser(truth_tensor_patch, net_input_patch, ref_t_patch, Phi_tensor_patch, 
                                                                    y_tensor_patch, model,optimizer, loss_fn, 
                                                                    dip_iter,y_w, tv_w,device)
                    
                    out_list_per_batch_size.append(torch.from_numpy(out_per_patch[:, :, padding_size : - padding_size, padding_size : - padding_size]).to(x.device))
                    loss_y_per_patch_size += loss_y_per_patch
                    
            loss_y_list.append(loss_y_per_patch_size)
                
            out_inner_list_per_patch_size = []
            
            for patch_w_id in range(patch_num):
                patch_w_start_id = patch_w_id * patch_num
                out_inner = out_list_per_batch_size[patch_w_start_id] 
                for patch_h_id in range(1, patch_num):
                    out_inner = torch.cat((out_inner, out_list_per_batch_size[patch_w_start_id + patch_h_id]), 3)
                out_inner_list_per_patch_size.append(out_inner)
            out = out_inner_list_per_patch_size[0]
            for out_inner_id in range(1, len(out_inner_list_per_patch_size)):
                out = torch.cat((out, out_inner_list_per_patch_size[out_inner_id]), 2)
                # out = torch.clamp(out,0,1)
            # out = out[:, 0::4, :, :]
            out_list.append(out)
            psnr_per_patch_size = psnr_torch(truth_tensor, out.squeeze(0))
            ssim_per_patch_size = ssim_torch(truth_tensor, out.squeeze(0))
            logger.info('Patch size {}, y_loss_per_patch_size = {:.5f}, PSNR_per_patch_size = {:.4f}, SSIM_per_patch_size = {:.4f}'.format(patch_size, loss_y_per_patch_size, psnr_per_patch_size,ssim_per_patch_size))
        
        out = torch.zeros(1, 8, 256, 256).to(x.device)
        loss_y_iter = 0
        for patch_size_id in range(len(patch_list)):
            out += out_list[patch_size_id]
            loss_y_iter += loss_y_list[patch_size_id]
        out /= len(patch_list)
        loss_y_iter /= len(patch_list)

        logger.info('Iteration {}, PSNR DIP = {:.4f}, SSIM DIP = {:.4f}'.format(it_dip_outer, psnr_torch(truth_tensor, out.squeeze(0)),ssim_torch(truth_tensor, out.squeeze(0))))

        # x_rec = torch.from_numpy(x_rec).to(x.device)
        # print(x_rec.shape)
        
        x = einops.rearrange(out,"1 f h w->1 f h w 1",f=frames)
        # x = x.view(batch_size,frames,height,width)
        if color_denoiser:
            src_x[0] = x[0,:,0::2,0::2,0] 
            src_x[1] = x[0,:,0::2,1::2,1] 
            src_x[2] = x[0,:,1::2,0::2,1] 
            src_x[3] = x[0,:,1::2,1::2,2]
            x = src_x 
        else:
            x = einops.reduce(x,"b f h w c->b f h w","max")
    
    x=weight*ref_t + (1-weight)*x
    # truth_tensor_psnr = torch.from_numpy(x_ori).to(x.device)
    x_psnr = einops.rearrange(x,"1 f h w->h w f").detach().cpu().numpy()
    psnr_x = psnr_block(x_ori, x_psnr)
    ssim_x = ssim_block(x_ori, x_psnr)
    lambda1 = lambda1*decrease
    return x,loss_y_iter,psnr_x,ssim_x, y1,lambda1