# image generation
from diffusers import StableDiffusionPipeline
import torch

# モデルの読み込み（軽量版を選択）
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 生成
prompt = "a cyberpunk cat in tokyo at night, neon lights, detailed"
image = pipe(prompt).images[0]

# 保存 & 表示
image.save("generated.png")
image.show()