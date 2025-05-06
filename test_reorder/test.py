import torch
import diffusers
from diffusers.pipelines.cogvideo.test_pipeline_cogvideox_image2video import test_CogVideoXImageToVideoPipeline
from diffusers.models.transformers.test_cogvideox_transformer_3d import test_CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
import torch.nn.functional as F
from sageattention import sageattn

diffusers.models.CogVideoXTransformer3DModel=test_CogVideoXTransformer3DModel

prompt = "A little boat sailing on the sea, realistic"
image = load_image(image="boat.jpg")
pipe = test_CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX1.5-5B-I2V",
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# 直接比较函数引用
if F.scaled_dot_product_attention is sageattn:
    print("正在使用 Sage Attention")
else:
    print("使用普通 Attention")

video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=81,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output_sage.mp4", fps=8)