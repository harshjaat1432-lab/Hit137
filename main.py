import os
import threading
import io
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import requests

# ----------------------------
# Configuration / Defaults
# ----------------------------
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"  # change if you prefer other HF text->image models
HF_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{}"
# ----------------------------

class HFTextToImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HuggingFace Text → Image (Tkinter)")
        self.geometry("820x700")
        self.resizable(False, False)

        # UI variables
        self.api_token_var = tk.StringVar(value=os.getenv("HF_API_TOKEN", ""))
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.prompt_var = tk.StringVar()
        self.width_var = tk.IntVar(value=512)
        self.height_var = tk.IntVar(value=512)
        self.steps_var = tk.IntVar(value=25)
        self.status_var = tk.StringVar(value="Ready")
        self.generated_image = None
        self.photo_image = None

        self._build_ui()

    def _build_ui(self):
        pad = 8
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=pad, pady=pad)

        # API token (small; optional)
        ttk.Label(frm_top, text="HF API Token:").grid(row=0, column=0, sticky="w")
        token_entry = ttk.Entry(frm_top, textvariable=self.api_token_var, width=64, show="*")
        token_entry.grid(row=0, column=1, sticky="w", padx=(6,0))
        ttk.Label(frm_top, text="(You can set HF_API_TOKEN env var instead)").grid(row=0, column=2, sticky="w", padx=(6,0))

        # Model
        ttk.Label(frm_top, text="Model:").grid(row=1, column=0, sticky="w", pady=(6,0))
        model_entry = ttk.Entry(frm_top, textvariable=self.model_var, width=64)
        model_entry.grid(row=1, column=1, sticky="w", padx=(6,0), pady=(6,0))

        # Prompt
        prompt_frame = ttk.Labelframe(self, text="Prompt")
        prompt_frame.pack(fill="x", padx=pad, pady=(0,pad))
        self.prompt_text = tk.Text(prompt_frame, height=4, width=96, wrap="word")
        self.prompt_text.pack(padx=6, pady=6)
        # Bind prompt_var not necessary; we'll read directly.

        # Parameters
        params_frame = ttk.Frame(self)
        params_frame.pack(fill="x", padx=pad, pady=(0,pad))

        ttk.Label(params_frame, text="Width:").grid(row=0, column=0, sticky="w")
        ttk.Entry(params_frame, textvariable=self.width_var, width=6).grid(row=0, column=1, sticky="w", padx=(4,12))
        ttk.Label(params_frame, text="Height:").grid(row=0, column=2, sticky="w")
        ttk.Entry(params_frame, textvariable=self.height_var, width=6).grid(row=0, column=3, sticky="w", padx=(4,12))
        ttk.Label(params_frame, text="Steps:").grid(row=0, column=4, sticky="w")
        ttk.Entry(params_frame, textvariable=self.steps_var, width=6).grid(row=0, column=5, sticky="w", padx=(4,12))

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=pad, pady=(0,pad))
        generate_btn = ttk.Button(btn_frame, text="Generate Image", command=self.on_generate_clicked)
        generate_btn.pack(side="left")
        clear_btn = ttk.Button(btn_frame, text="Clear Image", command=self.clear_image)
        clear_btn.pack(side="left", padx=(8,0))
        save_btn = ttk.Button(btn_frame, text="Save Image", command=self.save_image)
        save_btn.pack(side="left", padx=(8,0))

        # Status
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=pad)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")

        # Image display area
        disp_frame = ttk.Labelframe(self, text="Generated Image")
        disp_frame.pack(fill="both", expand=True, padx=pad, pady=pad)
        self.canvas = tk.Canvas(disp_frame, width=768, height=512, bg="#333333")
        self.canvas.pack(padx=6, pady=6)

    def set_status(self, text):
        self.status_var.set(text)
        self.update_idletasks()

    def on_generate_clicked(self):
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Prompt missing", "Please enter a text prompt to generate an image.")
            return

        token = self.api_token_var.get().strip() or os.getenv("HF_API_TOKEN")
        if not token:
            messagebox.showwarning("API token missing", "Provide a Hugging Face API token in the top field or set HF_API_TOKEN environment variable.")
            return

        # Read parameters
        model = self.model_var.get().strip() or DEFAULT_MODEL
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        steps = int(self.steps_var.get())

        # Start generation in a thread to avoid freezing UI
        t = threading.Thread(target=self.generate_image, args=(token, model, prompt, width, height, steps), daemon=True)
        t.start()

    def generate_image(self, token, model, prompt, width, height, steps):
        try:
            self.set_status("Preparing model / sending request...")
            api_url = HF_API_URL_TEMPLATE.format(model)

            headers = {
                "Authorization": f"Bearer {token}",
                # Request that the model return an image stream if available
                "Accept": "image/png"
            }

            payload = {
                "inputs": prompt,
                # Many HF text->image models accept "parameters" or model-specific fields.
                # We include common ones; some models ignore unknown parameters.
                "parameters": {
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps
                },
                # ask HF to load the model if it's not ready yet (can slow return)
                "options": {"wait_for_model": True}
            }

            # Make the request
            self.set_status("Contacting Hugging Face inference API (this may take a while)...")
            resp = requests.post(api_url, headers=headers, json=payload, timeout=600)

            # If HF returns an image directly, content-type starts with image/
            ctype = resp.headers.get("content-type", "")
            if resp.status_code == 200 and ctype.startswith("image/"):
                image_bytes = resp.content
            else:
                # HF may return JSON containing error or base64; try parse JSON
                try:
                    j = resp.json()
                except Exception:
                    j = None

                if isinstance(j, dict) and j.get("error"):
                    raise RuntimeError(f"Hugging Face API error: {j.get('error')}")
                # If returned something else, maybe the API responded with JSON image base64 or another structure.
                # Some models return an array of images as base64 strings: [{"generated_image": "<base64>"}] — handle common cases:
                # try to extract first base64 string anywhere in JSON
                import base64
                found_b64 = None
                def search_for_base64(obj):
                    nonlocal found_b64
                    if found_b64:
                        return
                    if isinstance(obj, str):
                        # Heuristic: base64 string has lots of letters and maybe slashes/plus and is long
                        if len(obj) > 200 and all(c.isalnum() or c in "+/=\n\r" for c in obj[:200]):
                            found_b64 = obj
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            search_for_base64(v)
                    elif isinstance(obj, list):
                        for v in obj:
                            search_for_base64(v)
                if j is not None:
                    search_for_base64(j)
                    if found_b64:
                        image_bytes = base64.b64decode(found_b64)
                    else:
                        raise RuntimeError(f"Unexpected API response (status {resp.status_code}): {j}")
                else:
                    raise RuntimeError(f"Unexpected API response (status {resp.status_code}) and not JSON. Content-type: {ctype}")

            # Load image into PIL
            img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            self.generated_image = img
            self.display_image(img)
            self.set_status("Image generated successfully.")
        except Exception as e:
            self.set_status("Error: " + str(e))
            messagebox.showerror("Generation error", str(e))

    def display_image(self, pil_img):
        # Resize to fit canvas while keeping aspect
        max_w, max_h = 768, 512
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Create a PhotoImage for Tkinter
        self.photo_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image((max_w//2, max_h//2), image=self.photo_image, anchor="center")

    def clear_image(self):
        self.canvas.delete("all")
        self.generated_image = None
        self.photo_image = None
        self.set_status("Cleared image.")

    def save_image(self):
        if not self.generated_image:
            messagebox.showinfo("No image", "No generated image to save.")
            return
        # Save to a file in current directory with a simple name
        filename = "hf_generated_image.png"
        try:
            self.generated_image.save(filename)
            messagebox.showinfo("Saved", f"Image saved as {filename}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

if __name__ == "__main__":
    app = HFTextToImageApp()
    app.mainloop()
