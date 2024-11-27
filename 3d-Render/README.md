# Simplest 3d-Render



## YBIGTA_DS 3D Assignment

Welcome to the 3D Assignment repository! This repository contains the basic rendering code to help you generate 3D rendered images. Your task is to execute the provided rendering code, obtain the resulting images, and write one-line explanations for specific rendering functions used in the code. Don't worry. No coding, 1 second inference time! (It's possible as this implementaion is most simplest, fixed pose with python.)



## Assignment Workflow

### Step 1: Understand the Code
1. Read the following files to familiarize yourself with their functionality:
   - `main.py`: Orchestrates the rendering process and calls necessary modules.
   - `objload.py`: Extracts information from `.obj` format files.
   - `render.py`: Performs one-direction rendering for the loaded 3D model.

2. Briefly understand how these components interact with each other.

---

### Step 2: Install Dependencies
Install the modules listed in `requirements.txt`. The installation process is dependency-independent:
```bash
pip install -r requirements.txt
```
---

### Step 3: Explore Example Files

Check out the provided example files in the `models` folder:

- `african_head.obj`, `LibertStatue.obj`: The 3D model file.
- `african_head.tga`: A texture file storing shading information.
- Rendered result examples:
	- `african_head_rendered_TGAo.bmp`
	- `african_head_rendered_TGAx.bmp`
    - `LibertStatue_rendered_TGAx.bmp`

You can learn `obj` file formats in <a href='https://en.wikipedia.org/wiki/Wavefront_.obj_file'>Wikipedia</a> and <a href='https://danac.tistory.com/155'>Blog(ko)</a>. Also, these files illustrate how `tga` files are used to store texture information during the rendering process. (Note: However, you can works without .tga files in this assignment for making our life better. It's hard to find a 3d model with `tga` file...)

---

### Step 4: Add Your Own Model

1. Download a free 3D model from a platform such as <a href='https://free3d.com/ko/3d-models/blender'>Free3D</a>.
2. Place your downloaded .obj and .tga files (if available) into the `models` folder.
3. Rename the files to match the naming convention and change the variable "model_name" correspondingly on `main.py`.

---

### Step 5: Run the Code
Run the provided `main.py` file to render your 3D model and generate a bitmap image:
```bash
python main.py
```
The rendered image will be output as a `.bmp` file.

---

### Step 6: Explain Rendering Functions

1. Open the file `explain_render_func.txt`.
2. Write one-line explanations for the rendering functions used in `render.py`. Briefly describe their role in the rendering process.

---

## Repository Structure
```
3d_Render/
├── models/                      # Folder containing 3D models and textures
│   ├── african_head.obj         # Example 1: 3D model
│   ├── african_head.tga         # Example 1: Texture file
│   ├── african_head_rendered_TGAo.bmp  # Example 1: Rendered output with texture
│   ├── african_head_rendered_TGAx.bmp  # Example 1: Rendered output without texture
│   ├── LibertStatue.obj         # Example 2: 3D model
│   ├── LibertStatue_rendered_TGAx.bmp  # Example 2: Rendered output without texture
│   ├── [your_model_files]       # Placeholder for your custom .obj and .tga files
│   └── [your_model_output]      # Placeholder for your rendered outputs
├── main.py                      # Main script to run the rendering process
├── objload.py                   # Script to load .obj files
├── render.py                    # Script for rendering 3D models
├── requirements.txt             # List of required Python modules
├── explain_render_func.txt      # Explanation of rendering functions
└── README.md                    # Documentation and usage instructions
```

---

## Notes

`.tga` files store texture information for the rendering process, but the rendering code is designed to function also without them.
Ensure your `.obj` and `.tga` files are compatible with the code.
