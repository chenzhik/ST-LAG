import torch
from diffusers import DDIMScheduler,PNDMScheduler, StableDiffusionPix2PixZeroPipeline, StableDiffusionPipeline
import requests, os
from PIL import Image
from diffusers.utils import load_image
import pickle
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import Raft_Small_Weights # raft_large
from animatediff.models.raft import raft_small_modified
from torchvision.utils import flow_to_image
from torchvision.models._utils import IntermediateLayerGetter
import numpy as np


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, height, width = imgs[0][0].shape
    th = num_rows * height
    tw = num_cols * width
    plots = np.zeros((th, tw, 3))
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            img = np.array(F.to_pil_image(img.to("cpu")))[:, :, ::-1]
            startw = col_idx * width 
            starth = row_idx * height
            plots[starth:starth+height, startw:startw+width, :] = img   
    return plots 

class FlowInterpolator:
    
    def __init__(self) -> None:
        weights= Raft_Small_Weights.C_T_V2
        self.raft_model = raft_small_modified(weights=weights, progress=False).cuda()
        # raft_model = raft_large(pretrained=True, progress=False).cuda()

        self.raft_model = self.raft_model.eval()
        self.totensor = T.ToTensor()
        self.topilimage = T.ToPILImage()
        self.transforms = T.Compose(
                [   
                    T.ConvertImageDtype(torch.float32),
                    T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                    # T.Resize(size=(520, 960)),
                ]
            )
    
    def estimate_flow(self, raw_images, exceed_size=False):
        """PIL image by load_image from diffusers.utils"""
        # max_size = 1024
        exceed_size = True
        if not exceed_size:
            # raw_tensor = [totensor(raw_image), for raw_image in raw_iamges]
            src_tensors = torch.stack([
                self.totensor(raw_images[i][0]) for i in range(0, len(raw_images)-1)
                ])
            tar_tensors = torch.stack([
                self.totensor(raw_images[i][0]) for i in range(1, len(raw_images))
            ])
            
            # estimate optical flow
            src_batch, tar_batch = self.transforms(src_tensors), self.transforms(tar_tensors)
            list_of_flows_small = self.raft_model.forward_wo_upsample(src_batch.cuda(), tar_batch.cuda())
            # image_flows = list_of_flows_large[-1]
            latent_flows = list_of_flows_small[-1]
            # flow_images = flow_to_image(image_flows)

        else:
            src_tensors = torch.stack([
                self.totensor(raw_images[i][0]) for i in range(0, len(raw_images)-1)
                ])
            tar_tensors = torch.stack([
                self.totensor(raw_images[i][0]) for i in range(1, len(raw_images))
            ])
            
            # estimate optical flow
            src_batch, tar_batch = self.transforms(src_tensors), self.transforms(tar_tensors)
            # latent_flows1 = self.raft_model.forward_wo_upsample(src_batch[:,:,0::2,0::2].cuda(), tar_batch[:,:,0::2,0::2].cuda())[-1]
            # latent_flows2 = self.raft_model.forward_wo_upsample(src_batch[:,:,0::2,0::2].cuda(), tar_batch[:,:,1::2,1::2].cuda())[-1]
            # latent_flows3 = self.raft_model.forward_wo_upsample(src_batch[:,:,1::2,1::2].cuda(), tar_batch[:,:,0::2,0::2].cuda())[-1]
            # latent_flows4 = self.raft_model.forward_wo_upsample(src_batch[:,:,1::2,1::2].cuda(), tar_batch[:,:,1::2,1::2].cuda())[-1]
            latent_flows = torch.zeros((15, 2, 256, 256)).cuda()
            latent_flows[:,:,0::2,0::2] = self.raft_model.forward_wo_upsample(src_batch[:,:,0::2,0::2].cuda(), tar_batch[:,:,0::2,0::2].cuda())[-1]
            latent_flows[:,:,1::2,1::2] = self.raft_model.forward_wo_upsample(src_batch[:,:,1::2,1::2].cuda(), tar_batch[:,:,1::2,1::2].cuda())[-1]
            latent_flows[:,:,1::2,0::2] = self.raft_model.forward_wo_upsample(src_batch[:,:,1::2,0::2].cuda(), tar_batch[:,:,1::2,0::2].cuda())[-1]
            latent_flows[:,:,0::2,1::2] = self.raft_model.forward_wo_upsample(src_batch[:,:,0::2,1::2].cuda(), tar_batch[:,:,0::2,1::2].cuda())[-1]
            
            # assert len(raw_images) == 16
            # length = max_length 
            # chunks = int((len(raw_images)-1) // (max_length+10e-3) + 1) # 15/2 = 8
            # latent_flows = []
            # for i in range(chunks): # 0-2, 2-4, 4-6, 6-8, 8-10, 10-12, 12-14, 14-15
            #     start_i, end_i = i*length, i*length+length # 0, 2
            #     if end_i + 1 > len(raw_images): end_i = len(raw_images) - 1
            #     src_tensors_clip = torch.stack([self.totensor(raw_images[i][0]) for i in range(start_i, end_i)]) # 0,1,2,3,4,5,6,7
            #     tar_tensors_clip = torch.stack([self.totensor(raw_images[i][0]) for i in range(start_i+1, end_i+1)]) # 1,2,3,4,5,6,7,8
            #     src_batch, tar_batch = self.transforms(src_tensors_clip), self.transforms(tar_tensors_clip)
            #     list_of_flows_small = self.raft_model.forward_wo_upsample(src_batch.cuda(), tar_batch.cuda())
            #     latent_flows.append(list_of_flows_small[-1])
            #     if end_i + 1 > len(raw_images): break
            # latent_flows = torch.cat(latent_flows, dim=0)
            
        return latent_flows
        
    def visualize_flow(self, src_batch_, tar_batch_, flow_images, warp_batch_):
        # visualize
        src_batch_ = [(img + 1) / 2 for img in src_batch]
        tar_batch_ = [(img + 1) / 2 for img in tar_batch]
        warp_batch_ = [(img + 1) / 2 for img in warp_batch]
        flow_grid = [[src_img, tar_img, flow_img, warp_img] for (src_img, tar_img, flow_img, warp_img) in zip(src_batch_, tar_batch_, flow_images, warp_batch_)]
        plots = plot(flow_grid)
        return plots
        
    def warping_with_flow(self, src_batch, tar_batch, flow, time=0):
        """ src: image or latent; flow: warp tar to src; time: 0(src) -> 1(tar)"""
        flow = (1 - time) * flow
        b, c, h, w = src_batch.shape
        grid =  torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')).unsqueeze(0).to(device=flow.device, dtype=flow.dtype) 
        grid = (flow + grid).permute(0, 2, 3, 1)
        grid = grid / torch.tensor([w - 1, h - 1], dtype=flow.dtype, device=flow.device) * 2. - 1.
        warped_src_from_tar = torch.nn.functional.grid_sample(tar_batch.cuda(), grid.to(torch.float16))
        return warped_src_from_tar
    
        
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    
    ##### optical model #####
    weights= Raft_Large_Weights.C_T_SKHT_V2
    raft_model = raft_modified(weights=weights, progress=False).cuda()
    # raft_model = raft_large(pretrained=True, progress=False).cuda()

    raft_model = raft_model.eval()
    totensor = T.ToTensor()
    topilimage = T.ToPILImage()
    transforms = T.Compose(
            [   
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                # T.Resize(size=(520, 960)),
            ]
        )
    
    #### diffusion model #####
    sd_model_ckpt = 'runwayml/stable-diffusion-v1-5'
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_ckpt,
        torch_dtype=torch.float16, 
        cache_dir="/mnt/afs/shuyan/.cache/huggingface/hub/",
        local_files_only=True)

    sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe.enable_model_cpu_offload()


    ##### prepare data #####
    videosamples = [
        "A-60-year-old-man-in-a-red-T-shirt-riding-a-trendy",
        "A-herd-of-camels-is-walking-on-the-golden-sand-dun",
        "A-man-taking-a-selfie-with-a-backdrop-of-hot-air-b",
        "An-Asian-person-wearing-a-waterproof-suit-fly-fish",
        "An-elderly-person-is-sitting-on-the-bed-and-taking"
        ]
    dire = "/mnt/afs_longfuchen/chenzhikai/project/Data/2023-10-28T00-03-40-eval-lf16-12000-ac-jug-kara/processed/frames/"
    df = pd.read_csv("/mnt/afs_longfuchen/chenzhikai/project/Data/image_juggernaut_prompts.11.23.csv")


    ##### run process #####
    for index, row in df.iterrows():
        path = row["image_paths"]
        videoname = path.split("/")[-1].replace(".png","").replace("\"","")
        if videoname in videosamples:
            prompt = row['captions']
            print(videoname, prompt)
            negative_prompt = ""

            raw_images = []
            for videoframe in range(16):
                framename = str(videoframe).zfill(8)
                raw_image = load_image(os.path.join(dire, videoname, framename + ".png"))
                raw_images.append(raw_image)

                caption = prompt
                
            # raw_tensor = [totensor(raw_image), for raw_image in raw_iamges]
            src_tensors = torch.stack([
                totensor(raw_images[0]), totensor(raw_images[1]), 
                totensor(raw_images[2]), totensor(raw_images[3])])
            tar_tensors = torch.stack([
                totensor(raw_images[1]), totensor(raw_images[2]), 
                totensor(raw_images[3]), totensor(raw_images[4])])
            
            # estimate optical flow
            src_batch, tar_batch = transforms(src_tensors), transforms(tar_tensors)
            list_of_flows_small, list_of_flows_large = raft_model.forward_wo_upsample(src_batch.cuda(), tar_batch.cuda())
            predicted_flows = list_of_flows_large[-1]
            flow_images = flow_to_image(predicted_flows)
            
            # warping
            b, c, h, w = src_batch.shape
            grid =  torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')).unsqueeze(0).to(device=predicted_flows.device, dtype=predicted_flows.dtype) 
            grid = (predicted_flows + grid).permute(0, 2, 3, 1)
            grid = grid / torch.tensor([w - 1, h - 1], dtype=predicted_flows.dtype, device=predicted_flows.device) * 2. - 1.
            warp_batch = torch.nn.functional.grid_sample(tar_batch.cuda(), grid)

            # visualize
            src_batch_ = [(img + 1) / 2 for img in src_batch]
            tar_batch_ = [(img + 1) / 2 for img in tar_batch]
            warp_batch_ = [(img + 1) / 2 for img in warp_batch]
            flow_grid = [[src_img, tar_img, flow_img, warp_img] for (src_img, tar_img, flow_img, warp_img) in zip(src_batch_, tar_batch_, flow_images, warp_batch_)]
            plots = plot(flow_grid)

            # predict latents
            latent_flows = list_of_flows_small[-1][0,:,:,:].unsqueeze(0) # [:,:,::8,::8]
            src_image = sd_pipe.image_processor.preprocess(raw_images[0]).to('cuda', torch.float16) 
            enc_latent_src = sd_pipe.vae.encode(src_image).latent_dist.sample() * sd_pipe.vae.config.scaling_factor
            tar_image = sd_pipe.image_processor.preprocess(raw_images[1]).to('cuda', torch.float16) 
            enc_latent_tar = sd_pipe.vae.encode(tar_image).latent_dist.sample() * sd_pipe.vae.config.scaling_factor

            # warping
            b, c, h, w = enc_latent_src.shape
            grid =  torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')).unsqueeze(0).to(device=latent_flows.device, dtype=latent_flows.dtype) 
            grid = (latent_flows + grid).permute(0, 2, 3, 1)
            grid = grid / torch.tensor([w - 1, h - 1], dtype=latent_flows.dtype, device=latent_flows.device) * 2. - 1.
            warp_latent = torch.nn.functional.grid_sample(enc_latent_tar.cuda(), grid.to(torch.float16))

            # visualize
            # dec_src_latent = 1 / sd_pipe.vae.config.scaling_factor * enc_latent_src
            # image = sd_pipe.vae.decode(dec_src_latent, return_dict=False)[0].detach()
            # rec_image = sd_pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=[True])

            dec_warp_latent = 1 / sd_pipe.vae.config.scaling_factor * warp_latent
            image_ = sd_pipe.vae.decode(dec_warp_latent, return_dict=False)[0].detach()
            warp_image = sd_pipe.image_processor.postprocess(image_, output_type='pil', do_denormalize=[True])
            warp_image[0].save(f"{videoname}_1_to_0.png")
