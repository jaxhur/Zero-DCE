import argparse
import csv
import math
import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import model


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def is_image_file(file_name):
	"""Return True when file_name has an image extension supported by this evaluator."""
	return os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS


def list_images(root_dir):
	"""Collect image paths below root_dir in a stable order."""
	image_paths = []
	for current_root, _, file_names in os.walk(root_dir):
		for file_name in file_names:
			if is_image_file(file_name):
				image_paths.append(os.path.join(current_root, file_name))
	return sorted(image_paths)


def relative_key(image_path, root_dir, keep_extension=True):
	"""Build a case-insensitive relative key used for LQ/GT pairing."""
	rel_path = os.path.relpath(image_path, root_dir)
	rel_path = rel_path.replace(os.sep, "/")
	if not keep_extension:
		rel_path = os.path.splitext(rel_path)[0]
	return rel_path.lower()


def build_image_pairs(low_dir, gt_dir):
	"""Pair low-light inputs with GT images by matching relative file names."""
	low_paths = list_images(low_dir)
	gt_paths = list_images(gt_dir)
	gt_by_exact_name = {relative_key(path, gt_dir): path for path in gt_paths}
	gt_by_stem = {relative_key(path, gt_dir, keep_extension=False): path for path in gt_paths}
	pairs = []
	missing = []

	for low_path in low_paths:
		exact_key = relative_key(low_path, low_dir)
		stem_key = relative_key(low_path, low_dir, keep_extension=False)
		gt_path = gt_by_exact_name.get(exact_key) or gt_by_stem.get(stem_key)
		if gt_path is None:
			missing.append(os.path.relpath(low_path, low_dir))
			continue
		pairs.append((low_path, gt_path, os.path.relpath(low_path, low_dir)))

	if missing:
		raise RuntimeError(
			"Missing GT images for {} low-light inputs. First missing item: {}".format(
				len(missing), missing[0]
			)
		)
	if not pairs:
		raise RuntimeError("No image pairs found. Check --low_dir and --gt_dir.")
	return pairs


def load_rgb_tensor(image_path, device):
	"""Load an RGB image as a BCHW float tensor in [0, 1]."""
	image = Image.open(image_path).convert("RGB")
	image_array = np.asarray(image).astype(np.float32) / 255.0
	image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
	return image_tensor.to(device)


def save_enhanced_image(enhanced_tensor, relative_path, output_dir):
	"""Save enhanced_tensor to output_dir while preserving the input relative name."""
	output_path = os.path.join(output_dir, relative_path)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	torchvision.utils.save_image(enhanced_tensor.clamp(0.0, 1.0), output_path)
	return output_path


def calculate_psnr(prediction, target):
	"""Calculate RGB PSNR for tensors in [0, 1]."""
	mse = torch.mean((prediction - target) ** 2).item()
	if mse == 0:
		return float("inf")
	return 20.0 * math.log10(1.0 / math.sqrt(mse))


def create_ssim_window(window_size, channel, device):
	"""Create the Gaussian window used by the RGB SSIM implementation."""
	coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
	gaussian = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
	gaussian = gaussian / gaussian.sum()
	window_2d = torch.outer(gaussian, gaussian)
	window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
	return window


def calculate_ssim(prediction, target, window_size=11):
	"""Calculate mean RGB SSIM for BCHW tensors in [0, 1]."""
	channel = prediction.size(1)
	window = create_ssim_window(window_size, channel, prediction.device)
	padding = window_size // 2
	c1 = 0.01 ** 2
	c2 = 0.03 ** 2

	mu_pred = F.conv2d(prediction, window, padding=padding, groups=channel)
	mu_target = F.conv2d(target, window, padding=padding, groups=channel)
	mu_pred_sq = mu_pred.pow(2)
	mu_target_sq = mu_target.pow(2)
	mu_pred_target = mu_pred * mu_target

	sigma_pred_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=channel) - mu_pred_sq
	sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu_target_sq
	sigma_pred_target = F.conv2d(prediction * target, window, padding=padding, groups=channel) - mu_pred_target

	ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
		(mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
	)
	return ssim_map.mean().item()


def build_lpips_model(lpips_net, device):
	"""Build the LPIPS model; the lpips package must be installed separately."""
	try:
		import lpips
	except ImportError as exc:
		raise ImportError("LPIPS evaluation requires `pip install lpips`.") from exc

	lpips_model = lpips.LPIPS(net=lpips_net).to(device)
	lpips_model.eval()
	return lpips_model


def calculate_lpips(prediction, target, lpips_model):
	"""Calculate LPIPS after mapping RGB tensors from [0, 1] to [-1, 1]."""
	prediction_lpips = prediction * 2.0 - 1.0
	target_lpips = target * 2.0 - 1.0
	with torch.no_grad():
		return lpips_model(prediction_lpips, target_lpips).item()


def parse_input_size(input_size_text):
	"""Parse an input-size string like 1,3,256,256 into a tuple of integers."""
	values = [int(value.strip()) for value in input_size_text.split(",")]
	if len(values) != 4:
		raise argparse.ArgumentTypeError("--flops_input_size must look like 1,3,256,256")
	return tuple(values)


def calculate_model_complexity(dce_net, input_size, device):
	"""Calculate Params(M), MACs(G), and FLOPs(G) with Conv2d forward hooks."""
	params_m = sum(parameter.numel() for parameter in dce_net.parameters()) / 1e6
	macs_total = 0
	hooks = []

	def conv_hook(module, inputs, output):
		nonlocal macs_total
		output_tensor = output[0] if isinstance(output, tuple) else output
		batch_size, out_channels, out_height, out_width = output_tensor.shape
		kernel_height, kernel_width = module.kernel_size
		kernel_ops = kernel_height * kernel_width * (module.in_channels // module.groups)
		macs_total += batch_size * out_channels * out_height * out_width * kernel_ops

	for module in dce_net.modules():
		if isinstance(module, nn.Conv2d):
			hooks.append(module.register_forward_hook(conv_hook))

	with torch.no_grad():
		dummy_input = torch.zeros(input_size, device=device)
		dce_net(dummy_input)

	for hook in hooks:
		hook.remove()

	macs_g = macs_total / 1e9
	flops_g = (2 * macs_total) / 1e9
	return params_m, macs_g, flops_g


def load_dce_net(weights_path, device):
	"""Load a Zero-DCE checkpoint into enhance_net_nopool."""
	dce_net = model.enhance_net_nopool().to(device)
	state_dict = torch.load(weights_path, map_location=device)
	if isinstance(state_dict, dict) and "state_dict" in state_dict:
		state_dict = state_dict["state_dict"]
	state_dict = {
		key.replace("module.", "", 1): value
		for key, value in state_dict.items()
	}
	dce_net.load_state_dict(state_dict)
	dce_net.eval()
	return dce_net


def write_per_image_csv(csv_path, rows):
	"""Write per-image metric rows to csv_path."""
	os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
	fieldnames = ["image", "psnr", "ssim", "lpips", "enhanced_path", "low_path", "gt_path"]
	with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def write_summary_csv(csv_path, summary):
	"""Write one summary row with average metrics and model complexity."""
	os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
	fieldnames = [
		"dataset",
		"num_images",
		"psnr",
		"ssim",
		"lpips",
		"params_m",
		"macs_g",
		"flops_g",
		"flops_input_size",
		"checkpoint",
		"low_dir",
		"gt_dir",
		"output_dir",
		"per_image_csv",
	]
	with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerow(summary)


def evaluate(config):
	"""Run Zero-DCE inference, save enhanced images, and write CSV metrics."""
	device = torch.device(config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu"))
	dce_net = load_dce_net(config.weights, device)
	input_size = parse_input_size(config.flops_input_size)
	params_m, macs_g, flops_g = calculate_model_complexity(dce_net, input_size, device)
	lpips_model = None if config.no_lpips else build_lpips_model(config.lpips_net, device)
	image_pairs = build_image_pairs(config.low_dir, config.gt_dir)
	per_image_rows = []
	psnr_values = []
	ssim_values = []
	lpips_values = []

	with torch.no_grad():
		for low_path, gt_path, relative_path in image_pairs:
			start_time = time.time()
			low_tensor = load_rgb_tensor(low_path, device)
			gt_tensor = load_rgb_tensor(gt_path, device)
			_, enhanced_tensor, _ = dce_net(low_tensor)
			enhanced_tensor = enhanced_tensor.clamp(0.0, 1.0)

			if enhanced_tensor.shape != gt_tensor.shape:
				raise RuntimeError(
					"Shape mismatch for {}: enhanced {} vs GT {}".format(
						relative_path, tuple(enhanced_tensor.shape), tuple(gt_tensor.shape)
					)
				)

			enhanced_path = save_enhanced_image(enhanced_tensor, relative_path, config.output_dir)
			psnr_value = calculate_psnr(enhanced_tensor, gt_tensor)
			ssim_value = calculate_ssim(enhanced_tensor, gt_tensor)
			lpips_value = None if lpips_model is None else calculate_lpips(enhanced_tensor, gt_tensor, lpips_model)
			elapsed_time = time.time() - start_time

			psnr_values.append(psnr_value)
			ssim_values.append(ssim_value)
			if lpips_value is not None:
				lpips_values.append(lpips_value)

			print(
				"{} | PSNR: {:.4f} | SSIM: {:.4f} | LPIPS: {} | {:.4f}s".format(
					relative_path,
					psnr_value,
					ssim_value,
					"{:.4f}".format(lpips_value) if lpips_value is not None else "N/A",
					elapsed_time,
				)
			)

			per_image_rows.append(
				{
					"image": relative_path,
					"psnr": "{:.6f}".format(psnr_value),
					"ssim": "{:.6f}".format(ssim_value),
					"lpips": "" if lpips_value is None else "{:.6f}".format(lpips_value),
					"enhanced_path": enhanced_path,
					"low_path": low_path,
					"gt_path": gt_path,
				}
			)

	per_image_csv = config.per_image_csv or os.path.join(config.output_dir, "per_image_metrics.csv")
	summary_csv = config.csv_path or os.path.join(config.output_dir, "summary_metrics.csv")
	write_per_image_csv(per_image_csv, per_image_rows)

	summary = {
		"dataset": config.dataset_name,
		"num_images": len(image_pairs),
		"psnr": "{:.6f}".format(float(np.mean(psnr_values))),
		"ssim": "{:.6f}".format(float(np.mean(ssim_values))),
		"lpips": "" if not lpips_values else "{:.6f}".format(float(np.mean(lpips_values))),
		"params_m": "{:.6f}".format(params_m),
		"macs_g": "{:.6f}".format(macs_g),
		"flops_g": "{:.6f}".format(flops_g),
		"flops_input_size": config.flops_input_size,
		"checkpoint": config.weights,
		"low_dir": config.low_dir,
		"gt_dir": config.gt_dir,
		"output_dir": config.output_dir,
		"per_image_csv": per_image_csv,
	}
	write_summary_csv(summary_csv, summary)
	print("Summary CSV saved to:", summary_csv)
	print("Per-image CSV saved to:", per_image_csv)
	print("Enhanced images saved to:", config.output_dir)


def parse_args():
	"""Parse command-line options for LOL paired evaluation."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--low_dir", type=str, required=True, help="Directory of low-light input images.")
	parser.add_argument("--gt_dir", type=str, required=True, help="Directory of paired normal-exposure GT images.")
	parser.add_argument("--weights", type=str, required=True, help="Path to the trained Zero-DCE checkpoint.")
	parser.add_argument("--output_dir", type=str, required=True, help="Directory for enhanced output images.")
	parser.add_argument("--dataset_name", type=str, default="LOL", help="Name written to the summary CSV.")
	parser.add_argument("--csv_path", type=str, default=None, help="Path of the summary CSV.")
	parser.add_argument("--per_image_csv", type=str, default=None, help="Path of the per-image CSV.")
	parser.add_argument("--device", type=str, default=None, help="Use cuda, cuda:0, or cpu. Default chooses CUDA if available.")
	parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone.")
	parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS when the lpips package is unavailable.")
	parser.add_argument("--flops_input_size", type=str, default="1,3,256,256", help="Input size used for FLOPs, e.g. 1,3,256,256.")
	return parser.parse_args()


if __name__ == "__main__":
	evaluate(parse_args())
