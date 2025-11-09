# cat_to_dog_inpaint
import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# --- Grad-CAM とマスク作成部分 ---
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_resnet18_gradcam_mask(img_pil, target_size=224, threshold=0.35):
    # 最新のAPI推奨形式でweightsを指定
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    # transformsをweightsから取得（mean/stdを自動で反映）
    preprocess = weights.transforms()

    input_tensor = preprocess(img_pil).unsqueeze(0)
    input_tensor.requires_grad_(True)

    # 最後の畳み込み層をhook
    final_conv = model.layer4[-1].conv2
    gradients = []
    activations = []

    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activation(module, input, output):
        activations.append(output)

    final_conv.register_forward_hook(save_activation)
    final_conv.register_full_backward_hook(save_gradient)

    # 推論
    output = model(input_tensor)
    pred_class = output.argmax().item()

    # 逆伝播（Grad-CAM）
    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0].mean(dim=(2, 3), keepdim=True)
    act = activations[0]
    cam = (act * grad).sum(dim=1).squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (target_size, target_size))
    cam = cam / cam.max()

    # マスク画像を生成
    mask = (cam > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).convert("L")

    return mask_img, pred_class, cam

# --- Inpainting 部分 (diffusers) ---
def inpaint_with_diffusers(original_pil, mask_pil, prompt,
                           output_path="inpaint_result.png",
                           model_repo = "stabilityai/sd-turbo-inpainting",
                           device=None):
    # lazy import to keep environment flexible
    from diffusers import StableDiffusionInpaintPipeline

    # device auto
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pipeline (may require huggingface auth; set HUGGINGFACE_TOKEN or login)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_repo,
        revision="fp16" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=True if os.getenv("HUGGINGFACE_TOKEN") else None,
    )
    pipe = pipe.to(device)

    # ensure sizes: many pipelines expect 512x512; resize if needed
    target_size = (512, 512)
    image_resized = ImageOps.fit(original_pil, target_size, method=Image.LANCZOS)
    mask_resized = ImageOps.fit(mask_pil.convert("L"), target_size, method=Image.NEAREST)

    # run inpaint
    with torch.autocast(device.type) if device.type == "cuda" else nullcontext():
        result = pipe(prompt=prompt, image=image_resized, mask_image=mask_resized).images[0]

    result.save(output_path)
    return result

# small helper for CPU context manager
from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield

# ---------------- main ----------------
if __name__ == "__main__":
    # --- 設定 ---
    img_path = r"C:\Users\narik\code\sample.png"  # 既に読み込めるPNGを想定
    prompt = "a realistic photo of a dog, photo-realistic, matching lighting and color of the original image"

    # --- load original ---
    orig = Image.open(img_path).convert("RGB")

    # --- Grad-CAM mask 作成 ---
    print("→ Grad-CAM を実行してマスクを作成します...")
    mask_img, pred_class, cam_map = get_resnet18_gradcam_mask(orig, target_size=224, threshold=0.35)
    mask_preview = Image.fromarray(np.array(mask_img).astype(np.uint8))
    mask_preview.save("debug_mask.png")
    print("  - mask saved to debug_mask.png")
    print("  - predicted class index (ResNet):", pred_class)

    # マスクを少し膨張（境界の余白を作る）したい場合は morphological 操作
    try:
        import cv2
        m = np.array(mask_img)
        kernel = np.ones((15,15), np.uint8)
        m = cv2.dilate(m, kernel, iterations=1)
        mask_img = Image.fromarray(m)
        mask_img.save("debug_mask_dilated.png")
        print("  - dilated mask saved to debug_mask_dilated.png")
    except Exception:
        print("  - OpenCV not installed; skipping dilation (optional).")

    # --- Inpaint ---
    print("→ Inpainting を実行します... (モデルロードに時間がかかります)")
    # 事前に HUGGINGFACE_TOKEN 環境変数を設定しておくと認証で困りにくいです
    result = inpaint_with_diffusers(orig, mask_img, prompt, output_path="cat_to_dog_result.png")
    print("→ 結果を cat_to_dog_result.png に保存しました。")
    result.show()
