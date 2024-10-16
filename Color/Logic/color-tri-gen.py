import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

# Function to generate a random triangle
def random_triangle(img_width, img_height):
    # Each triangle is defined by 3 points (x, y)
    triangle = [(random.randint(0, img_width), random.randint(0, img_height)) for _ in range(3)]
    return triangle

# Function to generate a random RGBA color with semi-transparency
def random_color():
    r = random.random()  # Red channel
    g = random.random()  # Green channel
    b = random.random()  # Blue channel
    a = random.uniform(0.3, 0.7)  # Alpha channel (semi-transparent)
    return (r, g, b, a)

# Load and resize the target image
def load_image(image_path, new_size=(200, 200)):
    image = Image.open(image_path).convert('RGB').resize(new_size)
    return np.array(image)

# Display input and generated images side by side
def display_images(target_image, generated_image, generation, fitness_score):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Show target image
    axs[0].imshow(target_image)
    axs[0].axis('off')
    axs[0].set_title('Target Image')

    # Show generated image
    axs[1].imshow(generated_image)
    axs[1].axis('off')
    axs[1].set_title(f'Generated Image (Gen {generation}, Fitness: {fitness_score})')

    plt.show()

# Example usage:
target_image = load_image(r"PATH")

class Triangle:
    def __init__(self, img_width, img_height):
        # Random points for the triangle
        self.points = random_triangle(img_width, img_height)
        # Random RGBA color for semi-transparency
        self.color = random_color()

class Individual:
    def __init__(self, num_triangles, img_width, img_height):
        self.triangles = [Triangle(img_width, img_height) for _ in range(num_triangles)]

# Increase to 100 triangles for detail
num_triangles = 100
img_width, img_height = target_image.shape[1], target_image.shape[0]
population_size = 200

# Initialize population
population = [Individual(num_triangles, img_width, img_height) for _ in range(population_size)]

def draw_individual(individual, img_width, img_height):
    # Create a figure with the size set to match the target image size (200x200 pixels)
    dpi = 100  # Set DPI (Dots Per Inch) to control resolution
    fig, ax = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.set_axis_off()

    for triangle in individual.triangles:
        poly = plt.Polygon(triangle.points, color=triangle.color)
        ax.add_patch(poly)

    fig.canvas.draw()

    # Convert the figure to a numpy array (RGBA)
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Now using RGBA

    plt.close(fig)  # Close the figure after drawing to avoid memory issues
    return img[:, :, :3]  # Return only RGB channels for fitness comparison

# Fitness function comparing RGB images
def fitness(individual, target_image):
    individual_img = draw_individual(individual, target_image.shape[1], target_image.shape[0])
    return np.sum(np.abs(target_image - individual_img))  # Sum of pixel differences

# Random crossover method
def random_crossover(parent1, parent2):
    child = Individual(len(parent1.triangles), img_width, img_height)
    for i in range(len(parent1.triangles)):
        if random.random() > 0.5:
            child.triangles[i] = parent1.triangles[i]
        else:
            child.triangles[i] = parent2.triangles[i]
    return child

# Subtle mutation with slower mutation rate decay
def mutate(individual, mutation_rate):
    for triangle in individual.triangles:
        if random.random() < mutation_rate:
            # Mutate the points of the triangle
            triangle.points = random_triangle(img_width, img_height)
            # Mutate the color with semi-transparency
            triangle.color = random_color()

# Slower mutation rate decay to keep more exploration throughout
def adaptive_mutation_rate(generation, total_generations):
    initial_rate = 0.4
    return initial_rate * np.exp(-generation / (total_generations / 2))

# Genetic algorithm with 500 generations and slower convergence
def genetic_algorithm(target_image, population_size=200, generations=100, milestone_improvement=100000):
    population = [Individual(num_triangles, img_width, img_height) for _ in range(population_size)]
    elite_size = int(population_size * 0.1)
    best_fitness = fitness(population[0], target_image)
    best_individual = population[0]

    for generation in range(generations):
        # Precompute fitness for all individuals
        fitness_values = [fitness(ind, target_image) for ind in population]

        # Adaptive mutation rate
        mutation_rate = adaptive_mutation_rate(generation, generations)

        # Selection and crossover
        new_population = population[:elite_size]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = random_crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)

        # Update population
        population = new_population

        # Track progress every 25 generations
        if generation % 25 == 0:
            best_gen_individual = min(population, key=lambda ind: fitness(ind, target_image))
            best_gen_fitness = fitness(best_gen_individual, target_image)
            print(f"Generation {generation}, Fitness = {best_gen_fitness}")

            if best_gen_fitness < best_fitness:
                best_fitness = best_gen_fitness
                best_individual = best_gen_individual

    print(f"Final Generation {generations}, Best Fitness = {best_fitness}")
    generated_image = draw_individual(best_individual, img_width, img_height)
    display_images(target_image, generated_image, generations, best_fitness)

    return best_individual

# Example usage
best_solution = genetic_algorithm(target_image, population_size=10, generations=100)
