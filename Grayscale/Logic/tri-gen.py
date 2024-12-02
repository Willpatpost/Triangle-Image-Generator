import argparse
import logging
import os
import random
import sys
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image
from skimage.filters import gaussian

# ================================== #
#            Configuration           #
# ================================== #

@dataclass
class Config:
    # Shape and Mutation Parameters
    mutation_rate: float = 1.0
    prob_add_del_element: float = 0.8  # Probability to add/delete elements
    prob_add_vs_del: float = 0.95      # Probability to add elements
    prob_exchange_elements: float = 0.1
    black_and_white: bool = True       # Use black and white shapes (grayscale)

    # Genetic Algorithm Parameters
    pop_size: int = 10
    nb_elements_max: int = 150         # Maximum number of elements
    nb_elements_initial: int = 10      # Initial number of elements
    nb_elite: int = 5
    element_transparency: float = -0.5  # <0 means use the shape's alpha
    background_colour: int = 255       # To be set to average color (grayscale)
    gaussian_blur_reference_image_sigma: float = 0.5
    new_shape_size_divisor: float = 4.0

    max_threads: int = 4
    optimization_partial_computation: bool = True

    # Shape Types (Only Triangle is used)
    SHAPE_TYPE_TRIANGLE: int = 1
    shape_type: int = 1  # Default to Triangle

    save_directory: str = "img/"

    # Logging Parameters
    enable_logging: bool = True
    log_interval: int = 10  # Seconds between logs
    log_level: str = "INFO"

    # Early Stopping Parameters
    stagnation_threshold: int = 5000  # Iterations without improvement
    fitness_goal: float = 0.001         # Fitness goal

# Initialize default configuration
config = Config()

# ================================== #
#          Utility Functions         #
# ================================== #

def clamp(val: int, a: int, b: int) -> int:
    """Clamp the value between a and b."""
    return max(a, min(val, b))

def normalize_fitness(fitness: float, size_x: int, size_y: int) -> float:
    """Normalize fitness by the number of pixels."""
    return fitness / (size_x * size_y)

def should_save_image(best_fitness: float, last_saved_fitness: float) -> bool:
    """Determine if the image should be saved based on fitness improvement."""
    # Save if fitness improved
    return best_fitness < last_saved_fitness

def setup_logging(log_level: str):
    """Configure the logging settings."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(
        level=numeric_level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Approximation using Triangles.")
    parser.add_argument('--iterations', type=int, default=30000, help='Number of evolutionary iterations.')
    parser.add_argument('--downsample', type=int, default=1, help='Factor by which to downsample the reference image for faster processing.')
    parser.add_argument('--black_and_white', action='store_true', help='Use black and white shapes.')
    args = parser.parse_args()
    return args

def ensure_save_directory():
    """Ensure the save directory exists, or create it."""
    if not os.path.exists(config.save_directory):
        try:
            os.makedirs(config.save_directory)
            if config.enable_logging:
                logging.info(f"Created save directory: {config.save_directory}")
        except Exception as e:
            logging.error(f"Failed to create save directory {config.save_directory}: {e}")
            sys.exit(1)

def read_img(filename: str) -> np.ndarray:
    """
    Read an image and convert it to a NumPy array.
    """
    try:
        img = Image.open(filename).convert('L')  # Ensure image is in grayscale
        ary = np.array(img)
        return ary
    except Exception as e:
        logging.error(f"Error reading image {filename}: {e}")
        sys.exit(1)

def save_bitmap(bitmap: np.ndarray, filename: str):
    """
    Save bitmap to a file.
    """
    try:
        bitmap = np.clip(bitmap, 0, 255).astype(np.uint8)
        im = Image.fromarray(bitmap, 'L')  # Save as grayscale
        im.save(filename)
        logging.info(f"Saved image: {filename}")
    except Exception as e:
        logging.error(f"Error saving image {filename}: {e}")

def create_gif(image_folder: str, gif_name: str):
    """
    Create a GIF from images in a folder.
    """
    try:
        images = []
        file_list = sorted(
            [f for f in os.listdir(image_folder) if f.endswith('.png')],
            key=lambda x: int(x.replace('iteration', '').replace('.png', '')) if 'iteration' in x else float('inf')
        )
        for filename in file_list:
            img = Image.open(os.path.join(image_folder, filename))
            images.append(img)
        if images:
            images[0].save(
                os.path.join(image_folder, gif_name),
                save_all=True,
                append_images=images[1:],
                duration=200,
                loop=0
            )
            logging.info(f"Created GIF: {os.path.join(image_folder, gif_name)}")
    except Exception as e:
        logging.error(f"Error creating GIF {gif_name}: {e}")

def initialize_logging():
    """Initialize logging based on configuration."""
    if config.enable_logging:
        setup_logging(config.log_level)
    else:
        # Disable logging by setting the level to CRITICAL
        logging.basicConfig(level=logging.CRITICAL)
    logging.debug("Logging initialized.")

def user_interaction():
    """Handle user interaction for logging options and image path."""
    # Logging options
    print("Choose logging option:")
    print("a) Default")
    print("b) Custom")
    print("c) None")
    print("d) Cancel program")
    logging_choice = input("Enter choice (a/b/c/d): ").strip().lower()

    if logging_choice == 'd':
        print("Program cancelled.")
        sys.exit(0)
    elif logging_choice == 'c':
        config.enable_logging = False
    elif logging_choice == 'b':
        custom_log_level = input("Enter custom logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL): ").strip().upper()
        if custom_log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config.log_level = custom_log_level
            config.enable_logging = True
        else:
            print("Invalid logging level. Using default.")
    else:
        config.enable_logging = True

    # File name prompt
    image_path = input("Enter desired target image file name: ").strip()
    while not os.path.isfile(image_path):
        print(f"File '{image_path}' not found. Please enter a valid file name.")
        image_path = input("Enter desired target image file name: ").strip()

    # Confirmation prompt
    print(f"The chosen file is '{image_path}'. Are you ready to begin?")
    print("a) Correct file, begin")
    print("b) Wrong file, re-enter")
    print("c) Wrong logging, return to beginning")
    print("d) Cancel program")
    confirmation_choice = input("Enter choice (a/b/c/d): ").strip().lower()

    if confirmation_choice == 'd':
        print("Program cancelled.")
        sys.exit(0)
    elif confirmation_choice == 'c':
        return user_interaction()  # Restart the user interaction
    elif confirmation_choice == 'b':
        return user_interaction()  # Restart the user interaction

    return image_path

def read_and_process_image(image_path: str, args):
    """Read and process the reference image."""
    # Read and process the reference image
    bitmap_reference = read_img(image_path)
    if args.downsample > 1:
        try:
            bitmap_reference = bitmap_reference[::args.downsample, ::args.downsample]
            if config.enable_logging:
                logging.info(f"Downsampled image by a factor of {args.downsample}. New size: {bitmap_reference.shape[1]}x{bitmap_reference.shape[0]}")
        except Exception as e:
            logging.error(f"Error downsampling image: {e}")
            sys.exit(1)

    # Ensure size_x is width and size_y is height
    canvas_height, canvas_width = bitmap_reference.shape
    if config.enable_logging:
        logging.info(f"Reference image size: {canvas_width}x{canvas_height} pixels.")

    # Apply Gaussian blur
    try:
        bitmap_reference = bitmap_reference.astype(float)
        bitmap_reference = gaussian(
            bitmap_reference,
            sigma=config.gaussian_blur_reference_image_sigma,
            channel_axis=None  # Since grayscale, no channel axis
        )
        bitmap_reference = np.clip(bitmap_reference, 0, 255).astype(int)
        if config.enable_logging:
            logging.info("Applied Gaussian blur to the reference image.")
    except Exception as e:
        logging.error(f"Error applying Gaussian blur: {e}")
        sys.exit(1)

    # Compute average color of the reference image and set it as the background color
    avg_color = int(np.mean(bitmap_reference))
    config.background_colour = avg_color
    if config.enable_logging:
        logging.info(f"Set background color to average color of the reference image: {avg_color}")

    return bitmap_reference, canvas_width, canvas_height

# ================================== #
#            Shape Classes           #
# ================================== #

class Shape:
    """Base Shape class."""
    def mutate(self):
        """Mutate the shape's attributes."""
        raise NotImplementedError

class Tri(Shape):
    """Triangle Shape."""
    def __init__(self, points: List[List[int]],
                 grayscale: int,
                 max_size_x: int, max_size_y: int, alpha: int):
        self.points = points  # [[pt0_x, pt0_y], [pt1_x, pt1_y], [pt2_x, pt2_y]]
        self.grayscale = grayscale
        self.alpha = clamp(alpha, 10, 245)
        self.max_size = [max_size_x, max_size_y]
        self.mask: Optional[np.ndarray] = None
        self.partial_computation: Optional[np.ndarray] = None

    def mutate(self):
        """Apply Gaussian mutation to the triangle's attributes."""
        # Apply Gaussian mutation to each point
        for i in range(3):
            for j in range(2):
                mutation = random.gauss(0, 1) * 10 * config.mutation_rate
                self.points[i][j] = clamp(
                    int(self.points[i][j] + mutation),
                    0, self.max_size[j] - 1
                )

        # Mutate grayscale value
        mutation = random.gauss(0, 1) * 10 * config.mutation_rate
        self.grayscale = clamp(
            int(self.grayscale + mutation),
            0, 255
        )

        # Mutate alpha
        mutation = random.gauss(0, 1) * 10 * config.mutation_rate
        self.alpha = clamp(
            int(self.alpha + mutation),
            10, 245
        )

        # Invalidate cached computations
        self.mask = None
        self.partial_computation = None

# ================================== #
#        ShapeImage Classes          #
# ================================== #

class ShapeImage:
    """Base ShapeImage class representing an image composed of multiple shapes."""
    def __init__(self, size_x: int, size_y: int, fixed_transparency: float):
        self.size = (size_x, size_y)
        self.lst_elements: List[Shape] = []
        self.fitness: float = -1.0
        self.fixed_transparency = fixed_transparency  # <0 means use the shape's alpha

    def copy_mutate(self) -> 'ShapeImage':
        """Create a duplicated and mutated copy of the image."""
        si = ShapeImage(self.size[0], self.size[1], config.element_transparency)
        si.lst_elements = copy.deepcopy(self.lst_elements)

        # Ensure a minimum number of shapes
        min_shapes = 5  # Set your desired minimum
        if len(si.lst_elements) < min_shapes:
            for _ in range(min_shapes - len(si.lst_elements)):
                si.add_random_element()

        # Attempt multiple mutations per copy to increase diversity
        for _ in range(2):  # Number of mutation attempts per copy
            operation = random.random()
            if operation < config.prob_add_del_element:
                if random.random() < config.prob_add_vs_del and len(si.lst_elements) < config.nb_elements_max:
                    si.add_random_element()
                elif len(si.lst_elements) > min_shapes:
                    idx_to_remove = random.randint(0, len(si.lst_elements) - 1)
                    del si.lst_elements[idx_to_remove]
                    logging.debug(f"Removed shape at index {idx_to_remove}")
            elif operation < config.prob_add_del_element + config.prob_exchange_elements and len(si.lst_elements) >= 2:
                # Exchange positions of two shapes
                idx1, idx2 = random.sample(range(len(si.lst_elements)), 2)
                si.lst_elements[idx1], si.lst_elements[idx2] = si.lst_elements[idx2], si.lst_elements[idx1]
                logging.debug(f"Exchanged shapes at indices {idx1} and {idx2}")
                for i in range(min(idx1, idx2), len(si.lst_elements)):
                    si.lst_elements[i].partial_computation = None
            elif len(si.lst_elements) > 0:
                # Mutate a random shape
                idx_elem_to_mutate = random.randint(0, len(si.lst_elements) - 1)
                si.lst_elements[idx_elem_to_mutate].mutate()
                logging.debug(f"Mutated shape at index {idx_elem_to_mutate}")
                for i in range(idx_elem_to_mutate, len(si.lst_elements)):
                    si.lst_elements[i].partial_computation = None

        return si

    def get_bitmap(self) -> np.ndarray:
        """Generate the bitmap image from shapes."""
        try:
            # Initialize bitmap
            ret_bmp = np.full((self.size[1], self.size[0]), config.background_colour, dtype=np.float32)

            for elem in self.lst_elements:
                if elem.partial_computation is not None:
                    ret_bmp = elem.partial_computation
                else:
                    mask = self.get_mask_matrix(elem)
                    if mask is None:
                        continue  # Skip if no mask

                    # Apply shape grayscale value where mask is True
                    shape_value = np.full((self.size[1], self.size[0]), elem.grayscale, dtype=np.float32)

                    # Handle transparency
                    if 0 <= self.fixed_transparency <= 1:
                        alpha_factor = self.fixed_transparency
                    else:
                        alpha_factor = elem.alpha / 255.0

                    # Blend the shape with the bitmap
                    ret_bmp = np.where(
                        mask,
                        alpha_factor * shape_value + (1.0 - alpha_factor) * ret_bmp,
                        ret_bmp
                    )

                    if config.optimization_partial_computation:
                        elem.partial_computation = ret_bmp.copy()

            return ret_bmp.astype(int)
        except Exception as e:
            logging.error(f"Exception in get_bitmap: {e}")
            raise

    def add_random_element(self):
        """Add a random Triangle to the list."""
        new_tri = Tri(
            points=[
                [random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)],
                [random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)],
                [random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)]
            ],
            grayscale=random.randint(0, 255),
            max_size_x=self.size[0],
            max_size_y=self.size[1],
            alpha=random.randint(10, 245)
        )
        self.lst_elements.append(new_tri)
        logging.debug(f"Added new triangle: Points={new_tri.points}, Grayscale={new_tri.grayscale}, Alpha={new_tri.alpha}")

    def get_mask_matrix(self, tri: Tri) -> Optional[np.ndarray]:
        """Generate a full-size mask matrix for a triangle."""
        try:
            if tri.mask is not None:
                return tri.mask

            # Create a grid of (x, y) coordinates
            x = np.arange(self.size[0])
            y = np.arange(self.size[1])
            xv, yv = np.meshgrid(x, y)

            # Extract triangle vertices
            pt0, pt1, pt2 = tri.points

            # Barycentric coordinate method
            det = (pt1[1] - pt2[1]) * (pt0[0] - pt2[0]) + (pt2[0] - pt1[0]) * (pt0[1] - pt2[1])
            if det == 0:
                tri.mask = np.zeros(self.size, dtype=bool)
                return tri.mask

            a = ((pt1[1] - pt2[1]) * (xv - pt2[0]) + (pt2[0] - pt1[0]) * (yv - pt2[1])) / det
            b = ((pt2[1] - pt0[1]) * (xv - pt2[0]) + (pt0[0] - pt2[0]) * (yv - pt2[1])) / det
            c = 1 - a - b

            mask = (a >= 0) & (b >= 0) & (c >= 0)
            tri.mask = mask
            return mask
        except Exception as e:
            logging.error(f"Exception in get_mask_matrix for Triangle: {e}")
            raise

# ================================== #
#        Genetic Algorithm        #
# ================================== #

def create_population(pop_size: int, elem_size: int, img_size_x: int, img_size_y: int) -> List[ShapeImage]:
    """
    Create the initial population of ShapeImages using Triangles.
    """
    pop = []
    for _ in range(pop_size):
        si = ShapeImage(img_size_x, img_size_y, config.element_transparency)
        for _ in range(elem_size):
            si.add_random_element()
        pop.append(si)
    logging.info(f"Initialized population with {pop_size} individuals.")
    return pop

def compare_bitmaps(bitmap1: np.ndarray, bitmap2: np.ndarray, size_x: int, size_y: int, num_shapes: int) -> float:
    """
    Compare two bitmaps and return the mean squared error (normalized), including a penalty for fewer shapes.
    """
    try:
        diff = bitmap1.astype(np.float32) - bitmap2.astype(np.float32)
        mse = np.mean(diff ** 2)
        mse_normalized = normalize_fitness(mse, size_x, size_y)
        # Introduce a penalty term inversely proportional to the number of shapes
        penalty = 1.0 / (num_shapes + 1)  # Add 1 to avoid division by zero
        return mse_normalized + penalty * 0.01  # Adjust the multiplier as needed
    except Exception as e:
        logging.error(f"Exception in compare_bitmaps: {e}")
        raise

def compare_pop_bitmap(bitmap_reference: np.ndarray, pop_member: ShapeImage, size_x: int, size_y: int):
    """Compare a population member's bitmap with the reference and update fitness."""
    try:
        if pop_member.fitness < 0:
            bitmap_pop = pop_member.get_bitmap()
            fitness = compare_bitmaps(bitmap_reference, bitmap_pop, size_x, size_y, len(pop_member.lst_elements))
            pop_member.fitness = fitness
            logging.debug(f"Computed fitness: {fitness:.4f}")
    except Exception as e:
        logging.error(f"Exception in compare_pop_bitmap: {e}")
        raise

def genetic_iteration_reproduce(pop: List[ShapeImage], nb_elite: int, bitmap_reference: np.ndarray) -> List[ShapeImage]:
    """
    Perform a genetic iteration using reproduction (crossover).
    """
    fit_results = get_fitness(pop, bitmap_reference)
    new_pop = []

    # Retain elite individuals
    for i in range(nb_elite):
        elite = pop[fit_results[i][0]]
        new_pop.append(copy.deepcopy(elite))

    # Perform crossover between elites
    for i in range(nb_elite):
        for j in range(nb_elite):
            if i != j and len(new_pop) < config.pop_size:
                offspring = pop[fit_results[i][0]].copy_mutate()
                new_pop.append(offspring)
                if len(new_pop) >= config.pop_size:
                    break
        if len(new_pop) >= config.pop_size:
            break

    logging.debug(f"Reproduced population size: {len(new_pop)}")
    return new_pop

def genetic_iteration_copy_mutate(pop: List[ShapeImage], nb_elite: int, bitmap_reference: np.ndarray, fit_results: List[tuple]) -> List[ShapeImage]:
    """
    Perform a genetic iteration using copy and mutate.
    """
    new_pop = []

    # Copy elite individuals
    for i in range(nb_elite):
        elite = copy.deepcopy(pop[fit_results[i][0]])
        new_pop.append(elite)

    # Mutate elite individuals
    for i in range(nb_elite):
        mutated_img = new_pop[i].copy_mutate()
        new_pop.append(mutated_img)

    # Fill the rest of the population with mutated individuals
    while len(new_pop) < config.pop_size:
        parent_idx = random.randint(0, nb_elite - 1)
        parent = pop[fit_results[parent_idx][0]]
        child = parent.copy_mutate()
        new_pop.append(child)

    logging.debug(f"Copy-mutate population size: {len(new_pop)}")
    return new_pop

def get_fitness(pop: List[ShapeImage], bitmap_reference: np.ndarray) -> List[tuple]:
    """Evaluate fitness for the entire population."""
    fit_results = []
    size_x, size_y = bitmap_reference.shape[1], bitmap_reference.shape[0]

    try:
        with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
            futures = [
                executor.submit(compare_pop_bitmap, bitmap_reference, pop[i], size_x, size_y)
                for i in range(len(pop))
            ]
            for future in futures:
                future.result()  # Wait for all to complete
    except Exception as e:
        logging.error(f"Error during fitness evaluation: {e}")
        raise

    for i, individual in enumerate(pop):
        fit_results.append((i, individual.fitness))

    fit_results.sort(key=lambda x: x[1])  # Sort by fitness (lower is better)
    logging.debug("Fitness evaluation completed successfully.")
    return fit_results

# ================================== #
#            Main Execution          #
# ================================== #

def main():
    """Main function to execute the genetic algorithm."""
    # User interaction
    image_path = user_interaction()

    # Parse arguments
    args = parse_arguments()

    # Update configuration based on arguments
    config.black_and_white = args.black_and_white

    # Initialize logging
    initialize_logging()
    if config.enable_logging:
        logging.info("Genetic Algorithm for Image Approximation Started.")

    # Create save directory
    ensure_save_directory()

    # Read and process the reference image
    bitmap_reference, canvas_width, canvas_height = read_and_process_image(image_path, args)

    # Create initial population
    initial_pop = create_population(config.pop_size, config.nb_elements_initial, canvas_width, canvas_height)
    logging.info("Initialized population with initial individuals.")

    # Evaluate fitness for the initial population
    try:
        logging.info("Starting fitness evaluation for the initial population.")
        fit_results = get_fitness(initial_pop, bitmap_reference)
        logging.info(f"Initial fitness results: {[f for f in fit_results]}")
    except Exception as e:
        logging.error(f"Error during initial fitness evaluation: {e}")
        sys.exit(1)

    pop = initial_pop
    last_best_fitness = float('inf')
    last_saved_fitness = float('inf')
    last_log_time = time.time()
    iterations_since_improvement = 0

    # Evolutionary Loop
    try:
        for i in range(args.iterations):
            logging.debug(f"Starting iteration {i}.")
            if i % 2 == 0:
                pop = genetic_iteration_reproduce(pop, config.nb_elite, bitmap_reference)
                logging.debug(f"Iteration {i}: Reproduction step completed.")
            else:
                pop = genetic_iteration_copy_mutate(pop, config.nb_elite, bitmap_reference, fit_results)
                logging.debug(f"Iteration {i}: Copy-Mutate step completed.")

            fit_results = get_fitness(pop, bitmap_reference)

            # Extract the best individual's index and fitness
            best_index, best_fitness = fit_results[0]
            num_shapes = len(pop[best_index].lst_elements)

            # Check for fitness improvement
            if best_fitness < last_best_fitness:
                iterations_since_improvement = 0
                last_best_fitness = best_fitness
                # Log the new best fitness
                logger_msg = f"New best fitness: {best_fitness:.4f} at iteration {i}!"
                if config.enable_logging:
                    logging.info(logger_msg)
            else:
                iterations_since_improvement += 1

            current_time = time.time()
            elapsed_time = current_time - last_log_time

            # Periodic Logging
            if elapsed_time >= config.log_interval:
                logger_msg = f"Iteration: {i}, Shapes: {num_shapes}, Best Fitness: {best_fitness:.4f}"
                if config.enable_logging:
                    logging.info(logger_msg)
                last_log_time = current_time

            # Save the best bitmap if fitness improved
            if should_save_image(best_fitness, last_saved_fitness):
                save_bitmap(pop[best_index].get_bitmap(), os.path.join(config.save_directory, f"iteration{i}.png"))
                last_saved_fitness = best_fitness

            # Early stopping
            if iterations_since_improvement >= config.stagnation_threshold:
                logger_msg = f"No improvement for {config.stagnation_threshold} iterations. Stopping evolution."
                if config.enable_logging:
                    logging.info(logger_msg)
                break

        logging.info("Evolution completed.")
    except KeyboardInterrupt:
        logging.warning("Evolution interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred during evolution: {e}")
        sys.exit(1)

    # Save Final Image
    final_best_index, final_best_fitness = fit_results[0]
    final_bitmap = pop[final_best_index].get_bitmap()
    save_bitmap(final_bitmap, os.path.join(config.save_directory, "final.png"))
    logging.info("Final image saved.")

    # Create GIF from saved images
    create_gif(config.save_directory, "evolution.gif")
    logging.info("GIF creation completed.")

if __name__ == "__main__":
    main()
