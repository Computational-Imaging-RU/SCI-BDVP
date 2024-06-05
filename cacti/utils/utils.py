import torch
import numpy as np
import math
import cv2
import os.path as osp
import einops
from cacti.utils.demosaic import demosaicing_CFA_Bayer_Menon2007 as demosaicing_bayer

def get_device_info():
    gpu_info_dict = {}
    if torch.cuda.is_available():
        gpu_info_dict["CUDA available"]=True
        gpu_num = torch.cuda.device_count()
        gpu_info_dict["GPU numbers"]=gpu_num
        infos = [{"GPU "+str(i):torch.cuda.get_device_name(i)} for i in range(gpu_num)]
        gpu_info_dict["GPU INFO"]=infos
    else:
        gpu_info_dict["CUDA_available"]=False
    return gpu_info_dict
    
def load_checkpoints(model,pretrained_dict,strict=False):
    # pretrained_dict = torch.load(checkpoints)
    if strict is True:
        try: 
            model.load_state_dict(pretrained_dict)
        except:
            print("load model error!")
    else:
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict: 
            if model_dict[k].shape != pretrained_dict[k].shape:
                pretrained_dict[k] = model_dict[k]
                print("layer: {} parameters size is not same!".format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict,strict=False)

def save_image(out,gt,image_name,show_flag=False):
    if len(out.shape)==4:
        out = einops.rearrange(out,"c f h w->h (f w) c")
        gt = einops.rearrange(gt,"c f h w->h (f w) c")
        result_img = np.concatenate([out,gt],axis=0)
        result_img = result_img[:,:,::-1]
    else:
        out = einops.rearrange(out,"f h w->h (f w)")
        gt = einops.rearrange(gt,"f h w->h (f w)")
        result_img = np.concatenate([out,gt],axis=0)
    result_img = result_img*255.
    cv2.imwrite(image_name,result_img)
    
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)
def save_single_image(images,image_dir,batch,name="",demosaic=False):
    images = images*255
    if len(images.shape)==4:
        frames = images.shape[1]
    else:
        frames = images.shape[0]
    for i in range(frames):
        begin_frame = batch*frames
        if len(images.shape)==4:
            single_image = images[:,i].transpose(1,2,0)[:,:,::-1]
        else:
            single_image = images[i]
        if demosaic:
            single_image = demosaicing_bayer(single_image,pattern='BGGR')
        cv2.imwrite(osp.join(image_dir,name+"_"+str(begin_frame+i+1)+".png"),single_image)
        
        
def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,dim=1,keepdim=True)
    return y

def At(y,Phi):
    x = y*Phi
    return x

def Avg(x,ii):
    x = np.mean(x[ii,:,:,:], axis=0)
    return x

def proj(v, p, x_avg, num_frames, zeta):
    scaled_x_avg = p * num_frames * x_avg.unsqueeze(0).unsqueeze(0)
    current_sum = v.sum(dim=1, keepdim=True)
    
    correction = (scaled_x_avg - current_sum) / num_frames
    
    v_projected = v + zeta*correction
    
    
    return v_projected

def A_torch(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At_torch(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

# def save_image(out,gt,image_name,show_flag=False):
#     sing_out = out.transpose(0,2,1).reshape(out.shape[1],-1)
#     if gt is None:
#         result_img = sing_out*255
#     else:
#         sing_gt = gt.transpose(0,2,1).reshape(gt.shape[1],-1)
#         result_img = np.concatenate([sing_gt,sing_out],axis=0)*255
#     result_img = result_img.astype(np.float32)
#     cv2.imwrite(image_name,result_img)
#     if show_flag:
#         cv2.namedWindow('image',0)
#         cv2.imshow('image',result_img.astype(np.uint8))
#         cv2.waitKey(0)



def psnr_block(ref, img):
    psnr = 0
    r,c,n = img.shape
    # PIXEL_MAX = ref.max()
    PIXEL_MAX = 1
    for i in range(n):
        mse = np.mean( (ref[:,:,i] - img[:,:,i]) ** 2 )
        psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr/n

def psnr_torch(ref, img):
    t,_,_ = ref.shape
    psnr = 0
    for i in range(t):
        mse = torch.mean( (ref[i,:,:] - img[i,:,:]) ** 2 )
        psnr += 20 * torch.log10(1 / mse.sqrt())
    return psnr/t