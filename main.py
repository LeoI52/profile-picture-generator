"""
@author : Léo Imbert
@created : 29/05/2025
@updated : 30/05/2025
"""

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from typing import Literal
import numpy as np
import opensimplex
import colorsys
import random


def get_palette_colors(number_colors:int, cmap_name:Literal['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'vanimo', 'hsv'])-> list:
    cmap = plt.get_cmap(cmap_name)
    return [tuple(int(c*255) for c in cmap(i/number_colors)[:3]) for i in range(number_colors)]


def generate_noise(width:int, height:int, octaves:float, seed:int, time_offset:float=0.0, show:bool=True)-> Image:
    opensimplex.seed(seed)
    img = Image.new("RGB", (width, height))

    for y in range(height):
        for x in range(width):
            value = opensimplex.noise2(x / octaves + time_offset, y / octaves + time_offset)

            normalized = (value + 1) / 2

            hue = normalized
            saturation = 0.7
            lightness = 0.5


            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            rgb = (int(r * 255), int(g * 255), int(b * 255))

            img.putpixel((x, y), rgb)

    if show:
        img.show()
    return img

def generate_grayscale_cellular_noise(width:int, height:int, number_cells:int, show:bool=True)-> Image:
    feature_points = [(random.randint(0, width), random.randint(0, height)) for _ in range(number_cells)]
    grid = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            distances = [((x - fx)**2 + (y - fy)**2)**0.5 for fx, fy in feature_points]
            grid[y, x] = min(distances)
    
    grid -= grid.min()
    grid /= grid.max()
    grid *= 255
    
    img = Image.fromarray(grid.astype(np.uint8), mode="L")
    if show:
        img.show()
    return img

def generate_colored_cellular_noise(width:int, height:int, number_cells:int, colors:list=None, show:bool=True)-> Image:
    feature_points = [(random.randint(0, width), random.randint(0, height)) for _ in range(number_cells)]
    colors = [(
        random.randint(100, 255),
        random.randint(100, 255),
        random.randint(100, 255)
    ) for _ in range(number_cells)] if not colors else colors

    img = Image.new("RGB", (width, height))
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            distances = []
            for i, (fx, fy) in enumerate(feature_points):
                dist = ((x - fx) ** 2 + (y - fy) ** 2) ** 0.5
                distances.append((dist, i))

            distances.sort()
            f1_dist, f1_idx = distances[0]
            f2_dist, f2_idx = distances[1]

            blend_factor = f1_dist / (f1_dist + f2_dist + 1e-5)
            blend_factor = blend_factor ** 0.7

            c1 = np.array(colors[f1_idx])
            c2 = np.array(colors[f2_idx])
            blended_color = (c1 * (1 - blend_factor) + c2 * blend_factor).astype(int)
            pixels[x, y] = tuple(blended_color)

    if show:
        img.show()
    return img

def generate_mandelbrot_fractal(width:int, height:int, max_iterations:int, center:tuple=(-0.5, 0), zoom:float=1.0, red_factor:float=2, green_factor:float=2, blue_factor:float=5, grayscale:bool=False, inverted:bool=False, show:bool=True)-> Image:
    zoom_factor = 1.5 / zoom
    xmin = center[0] - zoom_factor
    xmax = center[0] + zoom_factor
    ymin = center[1] - zoom_factor * (height / width)
    ymax = center[1] + zoom_factor * (height / width)

    mode = "RGB" if not grayscale else "L"
    img = Image.new(mode, (width, height))

    for y in range(height):
        for x in range(width):
            cx = (x * (xmax - xmin) / width + xmin)
            cy = (y * (ymin - ymax) / height + ymax)
            xn, yn, n = 0, 0, 0
            while (xn * xn + yn * yn) < 4 and n < max_iterations:
                xn, yn = xn * xn - yn * yn + cx, 2 * xn * yn + cy
                n += 1

            r, g, b = (red_factor * n) % 256, (green_factor * n) % 256, (blue_factor * n) % 256
            if inverted:
                r, g, b = 255 - r, 255 - g, 255 - b
            gray = (r + g + b) / 3

            if not grayscale:
                img.putpixel((x, y), (r, g, b))
            else:
                img.putpixel((x, y), int(gray))

    if show:
        img.show()
    return img

def generate_julia_fractal(width:int, height:int, max_iterations:int, red_factor:float=2, green_factor:float=2, blue_factor:float=5, grayscale:bool=False, inverted:bool=False, c:tuple=(0.285, 0.01), show:bool=True)-> Image:
    xmin, xmax, ymin, ymax = -1.25, 1.25, -1.25, 1.25
    mode = "RGB" if not grayscale else "L"
    img = Image.new(mode, (width, height))

    for y in range(height):
        for x in range(width):
            xn = (x * (xmax - xmin) / width + xmin)
            yn = (y * (ymin - ymax) / height + ymax)
            cx = c[0]
            cy = c[1]
            n = 0
            while (xn * xn + yn * yn) < 4 and n < max_iterations:
                xn, yn = xn * xn - yn * yn + cx, 2 * xn * yn + cy
                n += 1

            r, g, b = (red_factor * n) % 256, (green_factor * n) % 256, (blue_factor * n) % 256
            if inverted:
                r, g, b = 255 - r, 255 - g, 255 - b
            gray = (r + g + b) / 3

            if not grayscale:
                img.putpixel((x, y), (r, g, b))
            else:
                img.putpixel((x, y), int(gray))

    if show:
        img.show()
    return img

def generate_multibrot_fractal(width:int, height:int, max_iterations:int, power:int, center:tuple=(0, 0), zoom:float=1.0, red_factor:int=3, green_factor:int=5, blue_factor:int=7, grayscale:bool=False, inverted:bool=False, show:bool=True)-> Image:
    zoom_factor = 1.5 / zoom
    xmin = center[0] - zoom_factor
    xmax = center[0] + zoom_factor
    ymin = center[1] - zoom_factor * (height / width)
    ymax = center[1] + zoom_factor * (height / width)

    mode = "RGB" if not grayscale else "L"
    img = Image.new(mode, (width, height))

    for y in range(height):
        for x in range(width):
            cx = (x * (xmax - xmin) / width + xmin)
            cy = (y * (ymin - ymax) / height + ymax)
            z = complex(0, 0)
            c = complex(cx, cy)
            n = 0
            while abs(z) < 4 and n < max_iterations:
                z = z ** power + c
                n += 1

            r, g, b = (red_factor * n) % 256, (green_factor * n) % 256, (blue_factor * n) % 256
            if inverted:
                r, g, b = 255 - r, 255 - g, 255 - b
            gray = (r + g + b) // 3

            img.putpixel((x, y), (r, g, b) if not grayscale else int(gray))

    if show:
        img.show()
    return img


def generate_noise_gif(width:int, height:int, octaves:float, seed:int, number_frames:int, output_path:str):
    frames = []
    for i in range(number_frames):
        img = generate_noise(width, height, octaves, seed, i * 0.1, False)
        frames.append(img)
        print("\r", end="")
        filled_width = int(((i + 1) / number_frames) * 40)
        print(f"Frame n°{i + 1} [{"\033[32m█\033[00m"*filled_width}{"\033[31m▒\033[00m"*(40 - filled_width)}] {(i + 1) * 100 / number_frames:.1f}%", end="")

    print(f"\n\033[32mFile '{output_path}' successfully created !\033[00m")
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

def generate_mandelbrot_fractal_gif(width:int, height:int, max_iterations:int, number_frames:int, output_path:str, zoom_start:float=1.0, zoom_end:float=100.0, center:tuple=(-0.7435, 0.1314), red_factor:float=2, green_factor:float=2, blue_factor:float=5, grayscale:bool=False, inverted:bool=False):
    frames = []
    for i in range(number_frames):
        zoom = zoom_start * ((zoom_end / zoom_start) ** (i / (number_frames - 1)))
        img = generate_mandelbrot_fractal(width, height, max_iterations, center, zoom, red_factor, green_factor, blue_factor, grayscale, inverted, False)
        frames.append(img)
        print("\r", end="")
        filled_width = int(((i + 1) / number_frames) * 40)
        print(f"Frame n°{i + 1} [{"\033[32m█\033[00m"*filled_width}{"\033[31m▒\033[00m"*(40 - filled_width)}] {(i + 1) * 100 / number_frames:.1f}%", end="")

    print(f"\n\033[32mFile '{output_path}' successfully created !\033[00m")
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

def generate_multibrot_fractal_gif(width:int, height:int, max_iterations:int, power:int, number_frames:int, output_path:str, zoom_start:float=1.0, zoom_end:float=100.0, center:tuple=(-0.7435, 0.1314), red_factor:float=2, green_factor:float=2, blue_factor:float=5, grayscale:bool=False, inverted:bool=False):
    frames = []
    for i in range(number_frames):
        zoom = zoom_start * ((zoom_end / zoom_start) ** (i / (number_frames - 1)))
        img = generate_multibrot_fractal(width, height, max_iterations, power, center, zoom, red_factor, green_factor, blue_factor, grayscale, inverted, False)
        frames.append(img)
        print("\r", end="")
        filled_width = int(((i + 1) / number_frames) * 40)
        print(f"Frame n°{i + 1} [{"\033[32m█\033[00m"*filled_width}{"\033[31m▒\033[00m"*(40 - filled_width)}] {(i + 1) * 100 / number_frames:.1f}%", end="")

    print(f"\n\033[32mFile '{output_path}' successfully created !\033[00m")
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

def generate_julia_fractal_gif(width:int, height:int, max_iterations:int, number_frames:int, output_path:str, speed:float=0.01, red_factor:float=2, green_factor:float=2, blue_factor:float=5, grayscale:bool=False, inverted:bool=False, start_c:tuple=(0.285, 0.01)):
    frames = []
    for i in range(number_frames):
        img = generate_julia_fractal(width, height, max_iterations, red_factor, green_factor, blue_factor, grayscale, inverted, (start_c[0] + i * speed, start_c[1]), show=False)
        frames.append(img)
        print("\r", end="")
        filled_width = int(((i + 1) / number_frames) * 40)
        print(f"Frame n°{i + 1} [{"\033[32m█\033[00m"*filled_width}{"\033[31m▒\033[00m"*(40 - filled_width)}] {(i + 1) * 100 / number_frames:.1f}%", end="")

    print(f"\n\033[32mFile '{output_path}' successfully created !\033[00m")
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)


if __name__ == "__main__":
    pass