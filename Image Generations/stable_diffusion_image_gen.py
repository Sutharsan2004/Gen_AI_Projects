import torch
from diffusers import StableDiffusionXLPipeline
from compel import Compel, ReturnedEmbeddingsType
from IPython.display import display


pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)


pipe.enable_model_cpu_offload()


compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
    device="cuda"  # <--- THIS IS THE FIX
)

while True:
  inp = input("Enter prompt to generate image..")
  if inp.lower() == 'q':
    break
  
  conditioning, pooled = compel(inp)

  image = pipe(
    prompt_embeds=conditioning,
    pooled_prompt_embeds=pooled,
    num_inference_steps=40, 
    guidance_scale=7.0
  ).images[0]

  display(image)
  conf = input("Enter yes to download the image")
  if conf.lower() == "yes":
    file_name = input("Enter the name of the file")
    image.save(f"{file_name}.png")
