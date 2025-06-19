# Procedural Image Generator

Generate beautiful procedural art using noise functions, cellular algorithms, and fractals, also including animated GIFs.

## ✨ Features

This Python project allows you to generate stunning procedural images and animations using :
- Noise functions (OpenSimplex)
- Cellular textures (grayscale & color)
- Fractals: Mandelbrot, Julia, and Multibrot sets
- Animated GIFs with zoom or time-based progression

## 📸 Preview

![screenshot_1](/screenshot_1.png "Preview")

![screenshot_2](/screenshot_2.png "Preview")

![screenshot_3](/screenshot_3.gif "Preview")

## 🛠️ Requirements

- Python 3.7+
- Pillow
- matplotlib
- numpy
- opensimplex

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LeoI52/profile-picture-generator.git
   cd profile-picture-generator
   ```

2. **Install dependencies**:
   ```bash
   pip install Pillow matplotlib numpy opensimplex
   ```

## 📘 Usage

You can use it by importing it as a module and calling any generation function.
```Python
from filename import generate_mandelbrot_fractal

generate_mandelbrot_fractal(
    width=800,
    height=600,
    max_iterations=100,
    center=(-0.75, 0),
    zoom=1.0,
    show=True
)
```

## 🤝 Contributing

Contributions are welcome! If you have improvements or suggestions, feel free to fork the repo and submit a pull request.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](/LICENSE) for more information.