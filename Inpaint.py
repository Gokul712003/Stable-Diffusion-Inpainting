import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

init_image=Image.open('gautam.jpg')
mask_image=Image.open('mask.png')
prompt =' *Same Prompt HERE as if in Base Image Gen to outpaint the image for the aspect ratio* '
negativeprompt='cropped,worstquality'
steps=75
image_size=(720,1080)   #inpainting gives out 512x512 image Needed to be resized after inpainting to default image size inpainted

#sizes (1920,1080) for 16:9 (1440,1080) for 4:3 (720,1080) for 2:3
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",torch_dtype=torch.float16).to('cuda')

#pipe = StableDiffusionInpaintPipeline.from_pretrained(r"C:\Users\gokul\.cache\huggingface\hub\models--stabilityai--stable-diffusion-2-inpainting\snapshots\6ba40839c3c171123b2b863d16caf023e297abb9",torch_dtype=torch.float16).to('cuda')


pipe.enable_attention_slicing()


outpainted_image = pipe(prompt=prompt,negative_prompt=negativeprompt,image=init_image, mask_image=mask_image,num_inference_steps=steps,guidance_scale=7.5).images[0]
outpainted_image=outpainted_image.resize(image_size)
outpainted_image.save('Gautaminpainted.png')
