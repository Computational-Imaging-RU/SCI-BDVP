import os
import os.path as osp
import sys 
BASE_DIR=osp.dirname(osp.dirname(__file__))
sys.path.append(BASE_DIR)
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks,generate_mask
from cacti.utils.utils import At,save_single_image,get_device_info
from cacti.models.gap_denoise_dip_edge import GAP_denoise_dip
from cacti.models.gd_denoise_dip_edge import GD_denoise_dip
from cacti.datasets.builder import build_dataset 
from cacti.utils.logger import Logger
from cacti.utils.metrics import compare_psnr,compare_ssim
import torch
import numpy as np 
import argparse
import time 
import json 
import einops
import scipy.io as sio

def generate_meas(x,mask,device):
    temp = np.zeros_like(x[:,0,:,:])
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp[i,:,:] += x[i,j,:,:] * mask[j,:,:]
    temp = torch.from_numpy(temp)
    return temp.to(device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meas_noise",type=float, default=0.1)
    parser.add_argument("--denoise_method",type=str, default="GD_dip")
    parser.add_argument("--acc",type=bool,default=True)
    parser.add_argument("--step_size",type=float, default=0.1)
    parser.add_argument("--gradient_decrease_weight",type=float, default=1.0)
    parser.add_argument("--y_weight",type=float, default=0.1)
    parser.add_argument("--tv_weight",type=float, default=0.0)
    parser.add_argument("--lr",type=float, default=0.01)
    parser.add_argument("--noise_type",type=str, default='u')
    parser.add_argument("--kernelsize",type=int, default=3)
    parser.add_argument("--show_flag",type=bool, default=True)
    parser.add_argument("--TV_pre",type=bool, default=False)
    parser.add_argument("--TV_iter",type=int,default=30)
    parser.add_argument("--skip_weight",type=float, default=0.1)
    parser.add_argument("--data_aug",type=bool, default=False)
    parser.add_argument("--work_dir",type=str,default='./work_dirs')
    parser.add_argument("--mask_path",type=str,default='test_datasets/mask/binary_iid_mask_0.4.mat')
    parser.add_argument("--device",type=str,default="cuda:1")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device="cpu"
    return args

if __name__=="__main__":
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    test_data = dict(type="SixGraySimData",
                     data_root="test_datasets/simulation",
                     mask_path=args.mask_path,
                     mask_shape=None)
    # dip setup
    up_channel = [128,128,128,128]
    noise = [args.meas_noise]
    patch_list = [64,128,256]
    if args.meas_noise>0:
        if args.TV_pre:
            tv_iter = args.TV_iter
            dip_iter_list = [0,50,300,600,900,1500]
            outer_iter_list = [tv_iter,5,5,10,10,5]
        else:
            dip_iter_list =[50,300,600,900,1500]
            outer_iter_list = [5,5,10,10,5]
            outer_iter_list = [1]
    else:
        if args.TV_pre:
            tv_iter = args.TV_iter
            dip_iter_list = [0,75,400,1000,2000,4000]
            outer_iter_list = [tv_iter,5,10,20,20,20]
        else:
            dip_iter_list =[50,400,1000,2000,4000]
            outer_iter_list = [5,10,20,20,20]

    # create new folder
    out_path = os.path.join('./work_dirs', "_".join(map(str, [noise,args.denoise_method, args.acc, args.step_size, args.gradient_decrease_weight,
                                                                args.y_weight,args.tv_weight, args.noise_type,
                                                                args.lr, args.kernelsize, up_channel, patch_list, 
                                                                dip_iter_list,outer_iter_list,
                                                                args.skip_weight, 
                                                                args.show_flag,args.TV_pre,
                                                                args.TV_iter,args.mask_path])))
    args.work_dir = out_path
    log_dir = osp.join(args.work_dir,"log")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    logger.info(args)
    logger.info(args.mask_path)
    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    logger.info('GPU info:\n' + dash_line + env_info + '\n' + dash_line) 

    # load data
    mask, mask_s = generate_mask(test_data['mask_path'])
    test_data = build_dataset(test_data,{"mask":mask})
    data_loader = DataLoader(test_data,1)
    mask = torch.from_numpy(mask).to(device)
    frames,height,width = mask.shape
    mask_s = torch.from_numpy(mask_s).to(device)
    mask = einops.repeat(mask,"f h w-> b f h w",b=1)
    mask_s = einops.repeat(mask_s,"h w-> b f h w",b=1,f=1)
    
    # init evaluation metric
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    sum_time = 0.
    time_count = 0

    for add_para in noise:
        logger.info("addictive noise(sd): {}".format(add_para))
        for data_iter,data in enumerate(data_loader):
            psnr,ssim = 0,0
            batch_output = []
            _, gt = data
            gt = gt[0].cpu().numpy() # [batch_size,8,256,256]
            batch_size,frames,height,width = gt.shape
            mask_np = mask.cpu().numpy().squeeze(0)
            meas = generate_meas(gt,mask_np,device) # get meas from mask and gt
            add_noise=torch.randn(meas.size(),device=device) * add_para
            meas = meas + add_noise # addictive noise
            logger.info("data name: {}:".format(test_data.data_name_list[data_iter]))
            if add_para==0:
                if test_data.data_name_list[data_iter]=='crash32_cacti.mat':
                    outer_iter_list = [5,10,20]
                elif test_data.data_name_list[data_iter]=='aerial32_cacti.mat':
                    outer_iter_list = [5,10,20]
                elif test_data.data_name_list[data_iter]=='traffic_cacti.mat':
                    outer_iter_list = [5,10,20]
                else:
                    outer_iter_list = [5,10,20,20,20]
            for ii in range(batch_size):
                single_meas = meas[ii].unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 256, 256])
                y1 = torch.zeros_like(single_meas) 
                x = At(single_meas,mask)
                b = torch.zeros_like(x)
                v = torch.zeros_like(x)
                sum_iter = 0
                start = time.time()
                x_ori = gt[ii]
                x_ori = einops.rearrange(x_ori,"f h w->h w f")

                # SCI-BDVP-GAP
                if args.denoise_method =="GAP_dip":
                    loss_y_min = 1
                    y_w = args.y_weight
                    tv_w = args.tv_weight
                    step_size = args.step_size
                    up_channel = up_channel
                    dip_patch_list = patch_list
                    dip_iter_list = dip_iter_list
                    num = 0
                    eta = 1
                    lr = args.lr
                    TV_pre = args.TV_pre
                    TV_iter = args.TV_iter
                    outer_iter_list = outer_iter_list
                    acc = args.acc
                    aug = args.data_aug
                    kernel = args.kernelsize
                    noise_type = args.noise_type
                    weight = args.skip_weight
                    decrease = args.gradient_decrease_weight
                    for iter,iter_num in enumerate(outer_iter_list):
                        for it_dip in range(iter_num):
                            num += 1
                            dip_iter = dip_iter_list[iter]
                            x,loss_y_iter,psnr_x,ssim_x, y1,step_size = GAP_denoise_dip(x, single_meas,y1, mask, mask_s, dip_iter, x_ori, num, y_w, tv_w,step_size,dip_patch_list, lr, TV_pre,TV_iter,up_channel,kernel,aug,weight,noise_type,decrease,device,
                                                                       multichannel=True,accelerate=acc)
                            end = time.time()
                            logger.info('[{}/{}] PnP-DIP, Iteration {}, loss = {:.5f}, PSNR = {:.4f}, SSIM = {:.4f} time = {:.4f}'.format(ii+1,batch_size, num, loss_y_iter, psnr_x, ssim_x, (end-start)))

                # SCI-BDVP-PGD
                elif args.denoise_method =="GD_dip":
                    loss_y_min = 1
                    y_w = args.y_weight
                    tv_w = args.tv_weight
                    step_size = args.step_size
                    up_channel = up_channel
                    dip_patch_list = patch_list
                    dip_iter_list = dip_iter_list
                    num = 0
                    eta = 1
                    lr = args.lr
                    TV_pre = args.TV_pre
                    TV_iter = args.TV_iter
                    outer_iter_list = outer_iter_list
                    acc = args.acc
                    aug = args.data_aug
                    kernel = args.kernelsize
                    noise_type = args.noise_type
                    weight = args.skip_weight
                    decrease = args.gradient_decrease_weight
                    for iter,iter_num in enumerate(outer_iter_list):
                        for it_dip in range(iter_num):
                            num += 1
                            dip_iter = dip_iter_list[iter]
                            x,loss_y_iter,psnr_x,ssim_x, y1,step_size = GD_denoise_dip(x, single_meas,y1, mask, mask_s, dip_iter, x_ori, num, y_w, tv_w,step_size,dip_patch_list, lr, TV_pre,TV_iter,up_channel,kernel,aug,weight,noise_type,decrease,device,
                                                                       multichannel=True,accelerate=acc)
                            end = time.time()
                            logger.info('[{}/{}] PnP-DIP, Iteration {}, loss = {:.5f}, PSNR = {:.4f}, SSIM = {:.4f} time = {:.4f}'.format(ii+1,batch_size, num, loss_y_iter, psnr_x, ssim_x, (end-start)))       

                if ii>=0:
                    sum_time+=(end-start)
                    time_count += 1

                if args.show_flag:
                    logger.info(" ")
                output = x[0].cpu().numpy()
                batch_output.append(output)
                for jj in range(frames):
                    per_frame_out = output[jj]
                    per_frame_gt = gt[ii,jj, :, :]
                    psnr += compare_psnr(per_frame_gt*255,per_frame_out*255)
                    ssim += compare_ssim(per_frame_gt*255,per_frame_out*255)

                # calculate for one batch
                # break  
                
            psnr = psnr / (batch_size * frames)
            ssim = ssim / (batch_size * frames)
            logger.info("{}, Mean PSNR: {:.4f} Mean SSIM: {:.4f}.\n".format(
                        test_data.data_name_list[data_iter],psnr,ssim))
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            out_list.append(np.array(batch_output))

            # calculate for one image
            # break

        logger.info('Average Run Time:\n' 
                + dash_line + 
                "{:.4f} s.".format(sum_time/time_count) + '\n' +
                dash_line)

        test_dir = osp.join(args.work_dir,"test_images")
        if not osp.exists(test_dir):
            os.makedirs(test_dir)

        for i,name in enumerate(test_data.data_name_list):
            _name,_ = name.split("_")
            psnr_dict[_name] = psnr_list[i]
            ssim_dict[_name] = ssim_list[i]
            out = out_list[i]
            for j in range(out.shape[0]):
                image_dir = osp.join(test_dir,_name)
                if not osp.exists(image_dir):
                    os.makedirs(image_dir)
                save_single_image(out[j],image_dir,j)
        psnr_dict["psnr_mean"] = np.mean(psnr_list)
        ssim_dict["ssim_mean"] = np.mean(ssim_list)

        psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
        ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
        logger.info("Mean PSNR: \n"+
                    dash_line + 
                    "{}.\n".format(psnr_str)+
                    dash_line)

        logger.info("Mean SSIM: \n"+
                    dash_line + 
                    "{}.\n".format(ssim_str)+
                    dash_line)
    
