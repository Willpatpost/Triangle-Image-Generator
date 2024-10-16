import numpy as np
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim  # Import SSIM for fitness calculation

# Function to load and resize the target image as grayscale
def load_image(image_path, size=(200, 200)):
    image = Image.open(image_path).convert('L').resize(size)  # Grayscale and resize
    return np.array(image)

# Display input and generated images side by side
def display_images(target_image, generated_image, fitness_score):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Show target image
    axs[0].imshow(target_image, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Target Image')

    # Show generated image
    axs[1].imshow(generated_image, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(f'Best Generated Image (Fitness: {fitness_score})')

    plt.show()

# Create a random triangle with grayscale values
def random_triangle(img_width, img_height):
    points = [(random.randint(0, img_width), random.randint(0, img_height)) for _ in range(3)]
    grayscale = random.randint(0, 255)
    return points, grayscale

# Create an individual, which is a set of triangles
def create_individual(num_triangles, img_width, img_height):
    return [random_triangle(img_width, img_height) for _ in range(num_triangles)]

# Render an individual using numpy array manipulation directly (no matplotlib)
def render_individual(individual, img_width, img_height):
    img = Image.new('L', (img_width, img_height), color=0)  # Grayscale image with black background
    draw = ImageDraw.Draw(img)

    for points, grayscale in individual:
        draw.polygon(points, fill=int(grayscale))

    return np.array(img)

# Fitness function using SSIM (Structural Similarity Index)
def fitness_function(individual, target_image):
    generated_image = render_individual(individual, target_image.shape[1], target_image.shape[0])
    score, _ = ssim(target_image, generated_image, full=True)
    return score  # SSIM returns a value between -1 and 1, where 1 means perfect match

# Crossover between two parents to create a new individual
def crossover(parent1, parent2):
    child = []
    for tri1, tri2 in zip(parent1, parent2):
        if random.random() > 0.5:
            child.append(tri1)
        else:
            child.append(tri2)
    return child

# Mutate an individual by altering one of its triangles
def mutate(individual, mutation_rate, img_width, img_height):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random_triangle(img_width, img_height)

# Dynamic mutation rate that decreases over time to encourage exploration at the start
def dynamic_mutation_rate(generation, total_generations, initial_rate=0.05, final_rate=0.01):
    return initial_rate - (generation / total_generations) * (initial_rate - final_rate)

# Genetic algorithm to evolve a solution
def genetic_algorithm(target_image, num_triangles=500, population_size=1000, generations=200):
    img_width, img_height = target_image.shape[1], target_image.shape[0]

    # Initialize the population
    population = [create_individual(num_triangles, img_width, img_height) for _ in range(population_size)]
    best_fitness = -float('inf')  # SSIM closer to 1 is better, so start with a low value
    best_individual = None

    for generation in range(generations):
        # Evaluate fitness for each individual
        fitness_scores = [fitness_function(individual, target_image) for individual in population]

        # Find the best individual in this generation
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx]

        # Print the fitness score every 25 generations
        if generation % 25 == 0:
            print(f"Generation {generation}, Best Fitness (SSIM): {best_fitness}")

        # Dynamic mutation rate
        mutation_rate = dynamic_mutation_rate(generation, generations)

        # Selection (elite selection + crossover)
        elite_size = population_size // 2
        population = sorted(population, key=lambda ind: fitness_function(ind, target_image), reverse=True)[:elite_size]

        # Preserve the best individual
        population[0] = best_individual

        # Crossover and mutation to refill the population
        while len(population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate, img_width, img_height)
            population.append(child)

    # Final output: show the best solution
    best_image = render_individual(best_individual, img_width, img_height)
    display_images(target_image, best_image, best_fitness)
    print(f"Final Generation, Best Fitness (SSIM): {best_fitness}")

    return best_individual

# Example usage
target_image = load_image(r"C:\Users\Willp\Git\CS 580\Triangles\Senior Picture.JPG")
best_solution = genetic_algorithm(target_image, population_size=1000, generations=200)
