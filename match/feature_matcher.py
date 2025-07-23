import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import cv2
import faiss
import numpy as np
import torch
import torchvision
from PIL import Image

# import torchvision.transforms as T
# import transforms as T
from .dino_v2 import DinoV2Encoder


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0]
            > 0.5
        )

    return rescaled_image, target


__torchvision_need_compat_flag = float(torchvision.__version__.split(".")[1]) < 7
if __torchvision_need_compat_flag:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if __torchvision_need_compat_flag < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    # transform = T.Compose(
    #     [
    #         T.RandomResize([800], max_size=1333),
    #         T.ToTensor(),
    #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    # )
    image_source = Image.open(image_path)
    image_source = image_source.convert("RGBA")
    image = np.asarray(image_source)[:, :, :3]
    image_mask = np.asarray(image_source)[:, :, 3]
    # image_transformed, _ = transform(image_source, None)
    return image, image_mask


class FeatureMatcher(torch.nn.Module):
    ENCODERS = {
        "DinoV2Encoder": DinoV2Encoder,
        "DIFT": DinoV2Encoder,
        # "CLIPEncoder": CLIPEncoder,
    }

    def __init__(
        self,
        # Encoder kwargs
        encoder="DinoV2Encoder",
        encoder_kwargs=None,
        # Grounded SAM v2 kwargs
        gsam_box_threshold=0.25,  # Grounded SAM边界框的阈值
        gsam_text_threshold=0.25,  # Grounded SAM文本注释的阈值
        # General kwargs
        device="cuda",
        verbose=False,  # 是否打印详细信息
    ):
        """
        Args:
            encoder (str): Type of visual encoder used to generate visual embeddings.
                Valid options are {"DinoV2Encoder", "CLIPEncoder"}
            encoder_kwargs (None or dict): If specified, encoder-specific kwargs to pass to the encoder constructor
            gsam_box_threshold (float): Confidence threshold for generating GroundedSAM bounding box.
                If there are undetected objects in the input scene, consider decrease this value.
            gsam_text_threshold (float): Confidence threshold for generating GroundedSAM text annotation.
                If there are undetected objects in the input scene, consider decrease this value.
            device (str): Device to use for storing tensors
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        # Call super first
        super().__init__()

        # Initialize internal variables
        self.device = device
        self.verbose = verbose
        self.encoder_name = encoder

        # Create encoder # 创建编码器实例
        assert (
            encoder in self.ENCODERS
        ), f"Invalid encoder specified! Valid options are: {self.ENCODERS.keys()}, got: {encoder}"
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        encoder_kwargs["device"] = device
        self.encoder = self.ENCODERS[self.encoder_name](**encoder_kwargs)

        self.eval()

    def find_nearest_neighbor_candidates(
        self,
        input_img_fpath,  # 输入图像的绝对路径。
        candidate_imgs_fdirs=None,  # 候选图像目录路径。
        candidate_imgs=None,  # 候选图像的路径，如果提供了，直接使用这些图像。
        candidate_filter=None,  # 用于筛选候选图像的过滤器。
        n_candidates=2,  # 返回最相似的候选图像的数量。
        save_dir=None,  # 保存结果的目录路径。如果为 `None`，则使用输入图像所在的目录。
        visualize_resolution=(640, 480),
        boxes=None,  # 如果已计算，传入物体的边界框。
        logits=None,  # 如果已计算，传入分类的概率分布。
        phrases=None,  # 物体的描述词。
        obj_masks=None,  # 如果已计算，传入物体的分割掩码。
        save_prefix=None,  # 结果保存时的前缀。如果为 `None`，则使用 `input_category`。
        remove_background=True,  # 是否在计算 DINO 特征前移除背景。
        use_input_img_without_bbox=False,  # 是否直接使用输入图像计算特征，不使用边界框。
    ):
        """
        Args:
            input_category (str): Name of the desired object category to segment from
                @input_img_fpath. It is this category that is assumed will be attempted
                to be matched to nearest neighbor candidate(s)
            input_img_fpath (str): Absolute filepath to the input object image
            candidate_imgs_fdirs (None or str or list of str): Absolute filepath(s) to the candidate images directory(s)
            candidate_imgs (None or list of str): Absolute filepath(s) to the candidate images. If this is not None, directly use this. Otherwise, use candidate_imgs_fdirs
            candidate_filter (None or TextFilter): If specified, TextFilter for pruning all possible
                candidates from @candidate_imgs_fdir
            n_candidates (int): The number of nearest neighbor candidates to return.
            save_dir (None or str): If specified, the absolute path to the directory where the results should be saved.
                If None, will default to the same directory of @input_img_fpath.
            visualize_resolution (2-tuple): (H, W) when visualizing candidate results
            boxes (None or tensor): If specified, pre-computed SAM boxes to use
            logits (None or tensor): If specified, pre-computed SAM logits to use
            phrases (None or list): If specified, pre-computed SAM phrases to use
            obj_masks (None or np.array): If specified, pre-computed SAM segmentation mask to use
            save_prefix (None or str): If specified, the prefix string name for saved outputs.
                If None, saved outputs will be prepended with @input_category instead
            remove_background (bool): Whether to remove background before computing DINO features
            use_input_img_without_bbox (bool): Whether to directly use the input image to compute dino score,
                or with a bounding box of the target object

        Returns:
            dict: Dictionary of outputs. Note that this will also be saved to f"{save_prefix}_feature_matcher_results.json"
                in @save_dir
        """
        # 检查输入参数是否有效
        assert (candidate_imgs_fdirs is not None) or (candidate_imgs is not None)

        # Standardize save dir and make sure it exists
        save_dir = str(Path(input_img_fpath).parent) if save_dir is None else save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Standardize other inputs
        if candidate_imgs is None:
            candidate_imgs_fdirs = (
                [candidate_imgs_fdirs]
                if isinstance(candidate_imgs_fdirs, str)
                else candidate_imgs_fdirs
            )

        # Load the input image
        image_source, image_mask = load_image(input_img_fpath)
        ref_img_vis = cv2.resize(image_source, visualize_resolution)

        obj_mask = (image_mask > 0).astype(np.int8)

        # Mask the original image source and image and then infer the corresponding part-level segmentations
        # 使用分割掩码处理原始图像并进行特征提取
        # TODO: Black out all background pixels from ref_img_cropped using segmentation mask
        image_source_masked = (
            image_source
            * np.expand_dims(np.where(obj_mask > 0.5, 1.0, 0.0), axis=-1).astype(
                np.uint8
            )
            if remove_background
            else image_source
        )

        ref_img_cropped = image_source_masked
        obj_mask_cropped = obj_mask

        ref_img_masked_vis = cv2.resize(image_source_masked, visualize_resolution)

        # Get all valid candidates and load them # 加载所有候选图像
        models = (
            list(
                sorted(
                    f"{candidate_imgs_fdir}/{model}"
                    for candidate_imgs_fdir in candidate_imgs_fdirs
                    for model in os.listdir(candidate_imgs_fdir)
                    if (candidate_filter is None or candidate_filter.process(model))
                )
            )
            if candidate_imgs is None
            else sorted(candidate_imgs)
        )
        model_imgs = np.array(
            [np.array(Image.open(model).convert("RGB")) for model in models]
        )

        # 计算输入图像和候选图像的DINO特征，并将它们重塑为(N, D)的数组
        # Compute DINO features and reshape them to be (N, D) arrays
        ref_img_feats = self.encoder.get_features(ref_img_cropped).squeeze(
            axis=0
        )  # (84, 112, 384)
        model_imgs_feats = self.encoder.get_features(model_imgs)  # (64, 84, 112, 384)
        ref_feat_vecs = ref_img_feats.reshape(
            -1, self.encoder.embedding_dim
        )  # (9408, 384)

        # TODO: Remove background pixels from candidate feat vecs, via better dataset parsing (use alpha channel = 0.0)
        model_feat_vecs = model_imgs_feats.reshape(
            -1, self.encoder.embedding_dim
        )  # (602112, 384)

        # Reshape cropped image to be the same shape as the feature size
        if self.encoder_name == "DinoV2Encoder":
            H, W, C = ref_img_feats.shape
            obj_mask_cropped_resized = cv2.resize(
                obj_mask_cropped.astype(np.uint8), (W, H)
            )
        # elif self.encoder_name == "CLIPEncoder":
        #     obj_mask_cropped_resized = obj_mask_cropped
        else:
            raise ValueError(
                f"Got invalid encoder_name! Valid options: {self.ENCODERS.keys()}, got: {self.encoder_name}"
            )

        # 根据前景像素进行特征匹配
        # Get set of idxs corresponding to foreground
        foreground_idxs = set(obj_mask_cropped_resized.flatten().nonzero()[0])

        # Match features; compute top-K likely models
        top_k_models = []
        models_copy = deepcopy(models)
        model_imgs_copy = np.array(model_imgs)
        feat_vecs = np.array(model_feat_vecs)  # (602112, 384)
        # imgs = [ref_img_vis, ref_img_masked_vis]
        imgs = [ref_img_masked_vis]

        if self.verbose:
            print(
                f"{self.__class__.__name__}: Computing top-{n_candidates} candidates using encoder {self.encoder_name}..."
            )

        n_candidates = min(n_candidates, len(models))

        # 使用FAISS进行高效的特征匹配
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(self.encoder.embedding_dim)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        # 计算最接近的候选图像
        if self.encoder_name == "DinoV2Encoder":
            for i in range(n_candidates):
                if self.verbose:
                    print(
                        f"{self.__class__.__name__}: Computing DINO candidate {i+1}..."
                    )

                # 1
                # gpu_index_flat.reset()# 重置GPU索引，以便从头开始处理
                # gpu_index_flat.add(feat_vecs)# 将特征向量添加到索引中
                # # 使用faiss进行向量匹配，找到与参考图像特征最接近的一个特征向量
                # dists, idxs = gpu_index_flat.search(ref_feat_vecs, 1)   # (9408, 1) # 这里dists是距离，idxs是索引，表示找到的最相似的向量
                # # 将索引转换为对应的模型索引（通过除以图像的宽度和高度来缩放）
                # idxs = idxs // (H * W)
                # freqs = dict() # 用于记录每个模型的频率
                # 遍历找到的所有候选索引和距离
                # for j, (idx, dist) in enumerate(zip(idxs.reshape(-1), dists.reshape(-1))):
                #     # 如果该索引不属于前景区域，则跳过（即不考虑透明或背景像素）
                #     # If j is not part of foreground, skip
                #     if j not in foreground_idxs:
                #         continue

                #     # 如果该模型还没有记录过，则初始化频率
                #     if idx not in freqs:
                #         freqs[idx] = 0
                #     # Add weighting
                #     freqs[idx] += 1

                # 2
                freqs = dict()  # 用于记录每个模型的频率
                for idx in range(len(models)):
                    gpu_index_flat.reset()  # 重置GPU索引，以便从头开始处理
                    feat = feat_vecs[idx * (H * W) : (idx + 1) * (H * W)]
                    gpu_index_flat.add(feat)  # 将特征向量添加到索引中
                    # 使用faiss进行向量匹配，找到与参考图像特征最接近的一个特征向量
                    dists, _ = gpu_index_flat.search(
                        ref_feat_vecs, 1
                    )  # (9408, 1) # 这里dists是距离，idxs是索引，表示找到的最相似的向量
                    freqs[idx] = -dists.sum()

                # 根据频率对模型进行排序，按频率从低到高排序
                freqs = {
                    k: v for k, v in sorted(freqs.items(), key=lambda item: item[1])
                }
                # 选择出现频率最高的模型（即最匹配的模型）
                top_1_idx = list(freqs.keys())[-1]
                # 将最匹配的模型和图像加入到结果列表中
                top_k_models.append(models_copy[top_1_idx])
                imgs.append(
                    cv2.resize(model_imgs_copy[top_1_idx], visualize_resolution)
                )
                # Prune the selected one
                # 从候选列表中删除已经选择的模型（避免重复选择）
                del models_copy[top_1_idx]
                # 更新图像数组，删除已经选择的模型的图像
                model_imgs_copy = np.delete(model_imgs_copy, top_1_idx, axis=0)
                # 更新特征向量数组，删除已经选择的模型的特征向量
                feat_vecs = np.delete(
                    feat_vecs,
                    np.arange(H * W * top_1_idx, H * W * (top_1_idx + 1)),
                    axis=0,
                )

        elif self.encoder_name == "CLIPEncoder":
            gpu_index_flat.reset()
            gpu_index_flat.add(feat_vecs)
            dists, idxs = gpu_index_flat.search(ref_feat_vecs, n_candidates)

            # Store top-k models and distances
            for idx in idxs[0]:
                top_k_models.append(models_copy[idx])
                imgs.append(cv2.resize(model_imgs_copy[idx], visualize_resolution))

        # Record results # 记录结果并保存
        if self.verbose:
            print(
                f"Top-{n_candidates} models: {[model.split('/')[-1] for model in top_k_models]}"
            )
        concat_img = np.concatenate(imgs, axis=1)
        Image.fromarray(concat_img).save(
            f"{save_dir}/{save_prefix}_feature_matcher_results_visualization.png"
        )
        results = {
            "k": n_candidates,
            "candidates": top_k_models,
        }

        return results
