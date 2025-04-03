import torch
from telegram import Update, PhotoSize
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoencoderKL
import os
import io
import requests
from PIL import Image
import google.generativeai as genai
import time
import random


# --- Configuration ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Define Gemini model names
#GEMINI_TEXT_MODEL = 'gemini-1.5-flash-latest'
# GEMINI_VISION_MODEL = 'gemini-1.5-pro-latest'
GEMINI_TEXT_MODEL = 'gemini-2.0-flash-exp'
GEMINI_VISION_MODEL = 'gemini-2.0-flash-exp'

# System prompt for Gemini
SYSTEM_INSTRUCTION = ("You are a helpful, creative, clever, and very friendly AI assistant. "
                      "Respond conversationally. Your name is Mầm.")


if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_API_KEY environment variable not set.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
if not HF_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable not set.")

# Maximum history length
MAX_HISTORY = 10

class StableDiffusion(object):
    def __init__(   self,
                    model_id = "stabilityai/stable-diffusion-xl-base-1.0",
                    vae_id = "madebyollin/sdxl-vae-fp16-fix"
                ):
        self.pipe = None
        self.device = None
        self.dtype = None
        self.model_id = model_id
        self.vae_id = vae_id
        self.output = "generated_image.png"

    def setup(self):
        # Setup device, dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_dtype(None)
        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")
        self.setup_sdxl()

    def set_dtype(self, dtype=None):
        if dtype is None:
            if self.device == "cuda":
                # Check if BF16 is supported, otherwise use FP16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    self.dtype = torch.bfloat16
                    print("Using bfloat16 (requires Ampere GPU or newer)")
                else:
                    self.dtype = torch.float16
                    print("Using float16")
            else:
                self.dtype = torch.float32
                print("Using float32")
        else:
            self.dtype = dtype

    def setup_sdxl(self, model_id=None, vae_id=None):
        """Set up the Stable Diffusion XL pipeline."""
        if model_id is not None:
            self.model_id = model_id
        if vae_id is not None:
            self.vae_id = vae_id

        # --- Device and Data Type Setup ---
        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")

        # --- Load VAE (Optional but Recommended for SDXL) ---
        vae = None
        if self.vae_id and self.vae_id.lower() != 'none':
            try:
                print(f"Loading VAE: {self.vae_id}...")
                vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.dtype)
                print("VAE loaded.")
            except Exception as e:
                print(f"Warning: Could not load custom VAE {self.vae_id}: {e}. Using default.")
                vae = None  # Fallback to default VAE included in pipeline

        # --- Load SDXL Pipeline ---
        print(f"Loading model: {self.model_id}...")
        # Using EulerDiscreteScheduler is common for good results with fewer steps
        scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            vae=vae,  # Use the loaded VAE if available
            scheduler=scheduler,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,  # Use fp16 weights if available and using fp16
            use_safetensors=True
        )
        print("Model loaded.")

        # --- Move to Device ---
        pipeline.to(self.device)
        print(f"Pipeline moved to {self.device}.")

        # --- Optional Optimizations ---
        # Enable memory-efficient attention (helpful for lower VRAM, often default now)
        try:
            import xformers.ops
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention.")
        except ImportError:
            try:
                pipeline.enable_sdp_attention()
                print("Enabled sdp attention.")
            except:
                pipeline.enable_attention_slicing()
                print("Enabled attention slicing.")
            print("Enabled memory efficient attention.")
        except Exception as e:
            print(f"Could not enable attention optimizations: {e}. Proceeding without.")

        self.pipe = pipeline

    def generate(self
                , prompt= "A random beautiful scene"
                , negative_prompt= "ugly, deformed, noisy, blurry, low quality, text, words, writing, watermark, signature"
                , steps= 30
                , cfg= 7.5
                , width= 1024
                , height= 1024
                , seed= None):

        # --- Setup Generator for Seed ---
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using seed: {seed}")
        else:
            generator = torch.Generator(device=self.device).manual_seed(random.randint(0, 2**32 - 1))

        # --- Image Generation ---
        print(f"\nGenerating image...")
        print(f"  Prompt: '{prompt}'")
        print(f"  Negative Prompt: '{negative_prompt}'")
        print(f"  Steps: {steps}, CFG: {cfg}, Size: {width}x{height}")

        start_time = time.time()
        try:
            # Use inference_mode for less memory usage and potentially faster speed
            with torch.inference_mode():
                image: Image.Image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=cfg,
                    num_inference_steps=steps,
                    generator=generator,
                    # num_images_per_prompt=1, # Default is 1
                ).images[0] # The output is a list of images, get the first one

            end_time = time.time()
            print(f"Image generated in {end_time - start_time:.2f} seconds.")

            # --- Save Image ---
            print(f"Saving image to {self.output}...")
            try:
                # Ensure output directory exists
                output_dir = os.path.dirname(self.output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created output directory: {output_dir}")

                image.save(self.output)
                print(f"Image saved successfully to {os.path.abspath(self.output)}")
            except Exception as e:
                print(f"Error saving image: {e}")

        except Exception as e:
            print(f"\nError during image generation: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging

class Bot(object):
    def __init__(self, token=TELEGRAM_TOKEN):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}/"
        self.telegram = self.setup_telegram()
        self.sd = StableDiffusion()
        self.llm_text = None
        self.llm_photo = None
        self.llm_multi = None

    def setup_telegram(self):
        self.telegram = Application.builder().token(self.token).build()
        # Add handlers
        self.telegram.add_handler(CommandHandler("start", self.start))
        self.telegram.add_handler(CommandHandler("generate", self.generate))  # New handler for Stable Diffusion
        self.telegram.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        self.telegram.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.telegram.add_error_handler(self.error_handler)
        return self.telegram

    def setup_llm(self):
        # Configure the Google Generative AI client
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Set up the GenerativeModel for text and photo
            if GEMINI_VISION_MODEL == GEMINI_TEXT_MODEL:
                # Multimodal model
                self.llm_multi = genai.GenerativeModel(GEMINI_TEXT_MODEL)
                self.llm_text = None
                self.llm_photo = None
            else:
                self.llm_multi = None
                self.llm_text = genai.GenerativeModel(GEMINI_TEXT_MODEL, system_instruction=SYSTEM_INSTRUCTION)
                self.llm_photo = genai.GenerativeModel(GEMINI_VISION_MODEL)
        except Exception as e:
            print(f"Error configuring Google AI SDK: {e}")
            exit(1)

    def run(self):
        # Configure llm
        # Configure the Google Generative AI client
        self.setup_llm()
        # Start the bot
        print("Bot is running and polling for updates...")
        self.telegram.run_polling()

    # Telegram Command handler ------------------------------------------------
    # /start
    async def start( self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Hello! I'm Mầm. Send me a message, a photo, or use /generate <prompt> to create an image with Stable Diffusion!"
        )
        context.chat_data.pop('chat_history', None)
    # /generate
    async def generate(self, update: Update, context: ContextTypes.DEFAULT_TYPE, add_history=False):
        """Handles the /generate command to create an image with Stable Diffusion."""
        start_time = time.time()
        chat_id = update.effective_chat.id
        user_input = " ".join(context.args) if context.args else "A random beautiful scene"
        print(f"[Chat {chat_id}] Generate request: {user_input}")

        # Add to history
        if add_history:
            self.add_history(context, user_content=f"/generate {user_input}")

        try:
            # Send request to Stable Diffusion WebUI API
            await update.message.reply_text("Generating image, please wait...")
            self.sd.generate(prompt=user_input)

            # Send image to Telegram
            with open(self.sd.output, 'rb') as photo:
                await update.message.reply_photo(photo=photo,
                                                 caption=f"Generated. Time taken: {time.time() - start_time:.2f}s")

            # Clean up
            os.remove(self.sd.output)

            # Add to history
            if add_history:
                if self.llm_multi is not None:
                    # self.add_history(context, user_content=[f"[Image generated for prompt: '{user_input}']",photo])
                    img = Image.open(self.sd.output)
                    self.add_history(context, user_content=["Describe this image in detail.", img])
                    response = await self.llm_multi.generate_content_async([f"[Image generated for prompt: '{user_input}']", img], stream=True,
                                                            request_options={"timeout": 120})
                    ai_reply = ""
                    async for chunk in response:
                        if hasattr(chunk, 'text'):
                            chunk_text = chunk.text
                            print(chunk_text, end="", flush=True)
                            ai_reply += chunk_text
                        else:
                            print(f"\n[Chat {chat_id}] Non-text chunk received: {chunk.parts}\n")
                    self.add_history(context, model_content=ai_reply)
                else:
                    self.add_history(context, model_content=f"[Image generated for prompt: '{user_input}']")

        except requests.exceptions.RequestException as e:
            print(f"[Chat {chat_id}] Stable Diffusion API error: {e}")
            await update.message.reply_text(f"Error generating image: {e}")
        except Exception as e:
            print(f"[Chat {chat_id}] Error processing image: {e}")
            await update.message.reply_text(f"An error occurred: {e}")

    # Handle text messages
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_message_text = update.message.text
        chat_id = update.effective_chat.id
        print(f"[Chat {chat_id}] User: {user_message_text}")

        self.add_history(context, user_content=user_message_text)
        history = self.get_history(context)

        try:
            if self.llm_multi is not None:
                # Multimodal model
                self.llm_multi = genai.GenerativeModel(GEMINI_TEXT_MODEL, system_instruction=SYSTEM_INSTRUCTION)
                chat_session = self.llm_multi.start_chat(history=history[:-1])
            else:
                # Text model
                self.llm_text = genai.GenerativeModel(GEMINI_TEXT_MODEL, system_instruction=SYSTEM_INSTRUCTION)
                chat_session = self.llm_text.start_chat(history=history[:-1])
            response = await chat_session.send_message_async(history[-1]['parts'], stream=True)

            ai_reply = ""
            async for chunk in response:
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    print(chunk_text, end="", flush=True)
                    ai_reply += chunk_text
                else:
                    print(f"\n[Chat {chat_id}] Non-text chunk received: {chunk.parts}\n")

            print()
            if ai_reply:
                self.add_history(context, model_content=ai_reply)
                await update.message.reply_text(ai_reply)
            else:
                print(f"[Chat {chat_id}] Warning: Empty AI reply received.")
                await update.message.reply_text("Sorry, I couldn't generate a response.")

        except Exception as e:
            print(f"[Chat {chat_id}] Error generating Gemini text response: {e}")
            await update.message.reply_text(f"Sorry, an error occurred: {e}")
            history.pop()

    # Helper functions

    # Handle photo messages
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        print(f"[Chat {chat_id}] Received photo.")
        prompt_text = update.message.caption or "Describe this image in detail."
        print(f"[Chat {chat_id}] Prompt: {prompt_text}")
        photos: list[PhotoSize] = update.message.photo
        file_id = photos[-1].file_id
        file = await context.bot.get_file(file_id)
        file_url = file.file_path

        try:
            response = requests.get(file_url)
            response.raise_for_status()
            image_bytes = io.BytesIO(response.content)
            img = Image.open(image_bytes)
            if self.llm_multi:
                self.add_history(context, user_content=[prompt_text, img])
                response = await self.llm_multi.generate_content_async([prompt_text, img], stream=True,
                                                            request_options={"timeout": 120})
            else:
                response = await self.llm_photo.generate_content_async([prompt_text, img], stream=True,
                                                            request_options={"timeout": 120})

            ai_reply = ""
            async for chunk in response:
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    print(chunk_text, end="", flush=True)
                    ai_reply += chunk_text
                else:
                    print(f"\n[Chat {chat_id}] Non-text chunk received: {chunk.parts}\n")
            print()

            if ai_reply:
                self.add_history(context, user_content=f"[User sent an image with prompt: '{prompt_text}']",
                               model_content=ai_reply)
                await update.message.reply_text(ai_reply)
            else:
                print(f"[Chat {chat_id}] Warning: Empty AI reply received for image.")
                await update.message.reply_text("Sorry, I couldn't describe the image.")

        except requests.exceptions.RequestException as e:
            await update.message.reply_text(f"Error downloading image: {e}")
        except Exception as e:
            await update.message.reply_text(f"An error occurred: {e}")

    # Helpter functions --------------------------------------------------------
    def get_history( self, context: ContextTypes.DEFAULT_TYPE) -> list:
        return context.chat_data.setdefault('chat_history', [])

    def add_history( self, context: ContextTypes.DEFAULT_TYPE, user_content=None, model_content=None):
        history = self.get_history(context)
        if user_content is not None:
            if isinstance(user_content, list):
                history.append({'role': 'user', 'parts': user_content})
            else:
                history.append({'role': 'user', 'parts': [user_content]})
        if model_content is not None:
            history.append({'role': 'model', 'parts': [model_content]})
        while len(history) > MAX_HISTORY * 2:
            history.pop(0)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print(f"Error: Update {update} caused error {context.error}")


if __name__ == "__main__":
    # Step1 - Set up the bot
    bot = Bot(TELEGRAM_TOKEN)
    # Step2 - Set up the Stable Diffusion model
    bot.sd.setup()
    # Step3 - Run the bot
    bot.run()