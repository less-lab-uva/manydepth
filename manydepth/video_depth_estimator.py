import json
import os
import matplotlib.pyplot as plt

import cv2
import numpy as np

import torch

from manydepth import datasets, networks
from .layers import transformation_from_parameters, disp_to_depth
import glob
import PIL.Image as pil
from torchvision import transforms


MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def load_intrinsics(intrinsics_file, orig_width, orig_height):
    with open(intrinsics_file, 'r') as f:
        camera = json.load(f)
    fx = camera['intrinsic']['fx']
    fy = camera['intrinsic']['fy']
    u0 = camera['intrinsic']['u0']
    v0 = camera['intrinsic']['v0']
    intrinsics = np.array([[fx, 0, u0, 0],
                           [0, fy, v0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]).astype(np.float32)
    intrinsics[0, :] /= orig_width
    intrinsics[1, :] /= orig_height * 0.75
    invK = torch.Tensor(np.linalg.pinv(intrinsics)).unsqueeze(0)
    K = torch.Tensor(intrinsics).unsqueeze(0)

    if torch.cuda.is_available():
        return K.cuda(), invK.cuda()

    return K, invK


def preprocess_image(image, orig_width, orig_height, resize_width, resize_height):
    crop_height = (orig_height * 3) // 4
    image = image.crop((0, 0, orig_width, crop_height))
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda()
    return image


def calculate_depth(model_path, input_folder, input_semantics_folder, intrinsics_file, output_folder,
                    save_depth_img=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", model_path)

    # Loading pretrained model
    print("   Loading pretrained encoder")
    encoder_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    encoder = networks.ResnetEncoderMatching(18, False,
                                             input_width=encoder_dict['width'],
                                             input_height=encoder_dict['height'],
                                             adaptive_bins=True,
                                             min_depth_bin=encoder_dict['min_depth_bin'],
                                             max_depth_bin=encoder_dict['max_depth_bin'],
                                             depth_binning='linear',
                                             num_depth_bins=96)

    filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    print("   Loading pose network")
    pose_enc_dict = torch.load(os.path.join(model_path, "pose_encoder.pth"),
                               map_location=device)
    pose_dec_dict = torch.load(os.path.join(model_path, "pose.pth"), map_location=device)

    pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                    num_frames_to_predict_for=2)

    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)

    min_depth_bin = encoder_dict.get('min_depth_bin')
    max_depth_bin = encoder_dict.get('max_depth_bin')

    # Setting states of networks
    encoder.eval()
    depth_decoder.eval()
    pose_enc.eval()
    pose_dec.eval()
    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
        pose_enc.cuda()
        pose_dec.cuda()

    # Begin new code
    if output_folder[-1] != '/':
        output_folder += '/'
    os.makedirs(output_folder, exist_ok=True)
    files = list(sorted(glob.glob(input_folder + '/*.png')))
    semantic_files = list(sorted(glob.glob(input_semantics_folder + '/*.png')))
    input_images = [pil.open(file).convert('RGB') for file in files]
    input_semantic_images = [cv2.imread(file) for file in semantic_files]
    orig_width, orig_height = input_images[0].size
    K, invK = load_intrinsics(intrinsics_file, orig_width, orig_height)
    cropped_original_size = ((orig_height * 3) // 4, orig_width)
    processed_images = [preprocess_image(image,
                                         orig_width,
                                         orig_height,
                                         resize_width=encoder_dict['width'],
                                         resize_height=encoder_dict['height'])
                        for image in input_images]
    side_crop = (orig_width * 3) // 32
    top_crop = (orig_height // 4)
    depths = []
    masked_depths = []
    with torch.no_grad():
        for frame_index, (source_image, input_image,\
                          target_pred_image, input_file) in\
                enumerate(zip(processed_images, processed_images[1:],
                          input_semantic_images[1:], files[1:])):
            # print('Frame index', frame_index)
            pose_inputs = [source_image, input_image]
            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
            axisangle, translation = pose_dec(pose_inputs)
            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
            # print(pose[0, 0:3, 3])
            output, lowest_cost, _ = encoder(current_image=input_image,
                                             lookup_images=source_image.unsqueeze(1),
                                             poses=pose.unsqueeze(1),
                                             K=K,
                                             invK=invK,
                                             min_depth_bin=encoder_dict['min_depth_bin'],
                                             max_depth_bin=encoder_dict['max_depth_bin'])
            output = depth_decoder(output)
            pred_disp, depth = disp_to_depth(output[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
            depth_resized = torch.nn.functional.interpolate(
                depth, cropped_original_size, mode="bilinear", align_corners=False)
            depth_resized = depth_resized.cpu().numpy()[:, 0].squeeze()
            # print('depth resized', depth_resized.shape)
            cropped_depth = depth_resized.squeeze()[top_crop:, side_crop:-side_crop]
            pred_mask = ~np.logical_or((target_pred_image == [180, 130, 70]).all(axis=2),
                                       (target_pred_image == [35, 142, 107]).all(axis=2))
            # cv2.imshow('pred_image', target_pred_image)
            # indices = pred_mask.astype(np.uint8)  # convert to an unsigned byte
            # indices *= 255
            # cv2.imshow('Pred Mask', indices)
            # cv2.waitKey(0)
            # print('cropped depth', cropped_depth.shape)
            # print('pred mask', pred_mask[top_crop:-top_crop, side_crop:-side_crop].shape)
            pred_median = np.median(cropped_depth[pred_mask[top_crop:-top_crop, side_crop:-side_crop]])
            # The 17.07 hard coded comes from the median value of the medians in the ground truth.
            # Results from ground truth:
            # median 17.070566156794353
            # mean 17.151962683746994
            # std 3.5822013299780986
            # print(pred_median)
            ratio = 17.07 / pred_median
            # print(ratio)
            depth_resized *= ratio
            file_name = input_file[input_file.rfind('/'):]
            file_name = file_name[:file_name.rfind('.')]
            np.save(output_folder + file_name + '.npy', depth_resized)
            depths.append(depth_resized)
            if save_depth_img:
                clipped_depth = np.copy(depth_resized)
                clipped_depth[clipped_depth > 255] = 255  # clip into bounds
                cv2.imwrite(output_folder + file_name + '_depth.png', clipped_depth)
    #         tmp_depth = np.full(depth_resized.shape, np.nan)
    #         speed_mask = np.logical_and(pred_mask,
    #                                     ~(target_pred_image == [0, 0, 142]).all(axis=2))
    #         tmp_depth[speed_mask[:-top_crop, :]] = depth_resized[speed_mask[:-top_crop, :]]
    #         masked_depths.append(tmp_depth)
    # diffs = [masked_depths[i + 1] - masked_depths[i] for i in range(len(masked_depths) - 1)]
    # median_diffs = []
    # sigmas = []
    # for diff in diffs:
    #     diff = np.reshape(diff, (-1, 1))
    #     # print('nan count', np.count_nonzero(np.isnan(diff)))
    #     diff = diff[~np.isnan(diff)]
    #     lower_quartile = np.percentile(diff, 25)
    #     upper_quartile = np.percentile(diff, 75)
    #     diff = diff[np.logical_and(diff > lower_quartile, diff < upper_quartile)]
    #     # plt.hist(diff)
    #     # plt.show()
    #     median_diffs.append(np.median(diff))
    #     sigmas.append(np.std(diff))
    #     # print(np.min(diff), np.max(diff), np.mean(diff), np.median(diff))
    # # print(np.mean(avg_diffs))
    # print(np.median(median_diffs))
    # print(np.mean(sigmas))


if __name__ == '__main__':
    # python -m manydepth.video_depth_estimator
    calculate_depth('/data/manydepth/CityScapes_MR/',
                    '/data/manydepth/berlin_000440_input',
                    '/data/manydepth/berlin_000440_input_segmented',
                    '/seg_data/data/cityscapes/camera/test/berlin/berlin_000440_000019_camera.json',
                    '/data/manydepth/berlin_000440_output',
                    True)
    # calculate_depth('/data/manydepth/CityScapes_MR/',
    #                 '/data/manydepth/berlin_000000_input',
    #                 '/data/manydepth/berlin_000000_input_segmented',
    #                 '/seg_data/data/cityscapes/camera/test/berlin/berlin_000000_000019_camera.json',
    #                 '/data/manydepth/berlin_000000_output',
    #                 True)