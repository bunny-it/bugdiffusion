from datetime import datetime
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.utils import executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Telegram bot token (replace with your actual token)
TOKEN = ''

# Initialize the bot and dispatcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Initialize the stable diffusion pipeline and models
controlnet_conditioning_scale = 0.5
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

# Define the start command handler
@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    await message.reply("Welcome to the BugDiffusion Telegram bot. Send an image with caption, get a cup of coffee and enjoy!")

# Define the text processing handler
@dp.message_handler(lambda message: message.text and not message.text.startswith('/'))
async def process_text(message: types.Message):
    try:
        # Extract the user's message
        message_text = message.text
        user_id = message.from_user.id

        # Send instructions to the user
#        await message.reply_text("Please send an image to process along with your prompt.")

    except Exception as e:
        logging.error(str(e))
        await message.reply_text('An error occurred. Please try again.')

# Define the image processing handler
@dp.message_handler(content_types=types.ContentTypes.PHOTO)
async def process_image(message: types.Message):
    try:
        # Extract the user's prompt
        message_text = message.caption if message.caption else ""

        # Download and preprocess the user's image
        user_id = message.from_user.id
        image_path = f"input_{user_id}.jpg"
        await message.photo[-1].download(image_path)
        
        user_image = load_image(image_path)
        user_image = np.array(user_image)
        user_image = cv2.Canny(user_image, 100, 200)
        user_image = user_image[:, :, None]
        user_image = np.concatenate([user_image, user_image, user_image], axis=2)
        user_image = Image.fromarray(user_image)

        # Generate images using stable diffusion
        images = pipe(
            message_text, negative_prompt=None, image=user_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

        # Save the generated image
        output_path = f"output_{user_id}.png"
        images[0].save(output_path)

        # Send the generated image back to the user
        with open(output_path, "rb") as img_file:
            await message.reply_photo(photo=img_file)

        # Clean up generated files
        prefix = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        os.rename(image_path, prefix+image_path)
        os.rename(output_path, prefix+image_path)

    except Exception as e:
#        logging.error(str(e))
        await message.reply_text('Please send an image with a caption.')

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
