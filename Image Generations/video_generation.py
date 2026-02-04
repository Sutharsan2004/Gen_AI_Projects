import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# 1. Load the pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload() # Saves VRAM

# 2. Define your prompt
prompt = "An astronaut walking on a neon-lit street in a futuristic city, cinematic style"

# 3. Generate the video frames
# num_frames=16 is standard for this model
video_frames = pipe(prompt, num_frames=16).frames[0]

# 4. Export to a playable MP4 file
video_path = export_to_video(video_frames, output_video_path="generated_video.mp4")

print(f"Video saved to: {video_path}")
