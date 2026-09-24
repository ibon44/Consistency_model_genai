import os
import random
import numpy as np
from PIL import Image

import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# Ruta donde se guardarán las imágenes RGB
output_dir = "./data/colored_mnist"
os.makedirs(output_dir, exist_ok=True)

# MNIST en blanco y negro
mnist = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Convertir y guardar cada imagen en RGB
for idx, (img, label) in enumerate(tqdm(mnist)):
    # Escala de grises → numpy [0,1]
    img_np = img.squeeze().numpy()

    # Generar color aleatorio (R,G,B) ∈ [0.5, 1.0] para visibilidad
    color = np.random.uniform(0.5, 1.0, size=(3, 1, 1))
    
    # Expandir a RGB y aplicar el color
    colored_img = img_np[None, ...] * color
    colored_img = np.clip(colored_img, 0, 1)

    # Convertir a imagen PIL
    colored_pil = Image.fromarray((colored_img.transpose(1, 2, 0) * 255).astype(np.uint8))

    # Guardar en carpeta por clase
    label_dir = os.path.join(output_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)
    img_path = os.path.join(label_dir, f"{idx}.png")
    colored_pil.save(img_path)
