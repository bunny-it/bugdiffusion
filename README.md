
# BugDiffusionBot
Telegram bot using stable diffusion on GPU to create images.

## Prerequisites

### Hardware

- A GPU with at least 12GB RAM
- 32GB of RAM


### Software
- Python <= 3.11
- python-pip
- miniconda

### Telegram Bot Token

- Create your bot on Telegram by contacting https://t.me/BotFather
- Copy the token and modify `TOKEN = ''` inside bugdiffusion.py

1. Create virtual environment:

`conda create -n bugdiffusion`

2. Load it:

`conda activate bugdiffusion`

3. Install requirements:

`pip install -r requirements.txt`

4. Run server:

`python bugdiffusion.py`


Now start by sending an image to your bot. If you want to include a prompt send the image with a caption.

Have fun!


__bunny__
