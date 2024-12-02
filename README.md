# Genetic Algorithm for Image Approximation Using Triangles

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)

## Overview

This project implements a **Genetic Algorithm (GA)** to approximate a given grayscale image using a set of semi-transparent triangles. Inspired by Roger Johansson's blog on evolving the Mona Lisa, the algorithm iteratively evolves a population of triangle-based images to closely resemble the target image. Each individual in the population consists of multiple triangles, and through processes like selection, crossover, and mutation, the GA optimizes the placement, size, grayscale value, and transparency of these triangles to minimize the difference from the target image.

## Features

- **Triangle-Based Image Representation**: Uses semi-transparent triangles to compose the approximated image.
- **Customizable Genetic Algorithm Parameters**: Adjust population size, mutation rates, crossover rates, and more.
- **Fitness Evaluation**: Measures similarity to the target image using normalized mean squared error.
- **Parallel Fitness Computation**: Utilizes multithreading to speed up fitness evaluations.
- **Image Processing**: Applies Gaussian blur to the target image for smoother fitness calculations.
- **Output Visualization**: Saves intermediate images and compiles them into an evolution GIF to visualize the optimization process.
- **Logging**: Provides detailed logs for monitoring the algorithm's progress and debugging.

## Requirements

- **Python 3.7 or higher**

### Python Libraries

- [NumPy](https://numpy.org/) (`numpy`)
- [Pillow](https://python-pillow.org/) (`Pillow`)
- [scikit-image](https://scikit-image.org/) (`scikit-image`)

### Installation of Python Libraries

You can install the required Python libraries using `pip`:

```bash
pip install numpy Pillow scikit-image
