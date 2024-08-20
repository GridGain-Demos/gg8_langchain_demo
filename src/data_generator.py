import random
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Components
cpus = {
    "low": ["Intel i3", "Intel i5", "AMD Ryzen 3", "AMD Ryzen 5"],
    "medium": ["Intel i5", "Intel i7", "AMD Ryzen 5", "AMD Ryzen 7"],
    "high": ["Intel i7", "Intel i9", "AMD Ryzen 7", "AMD Ryzen 9"]
}
gpus = ["Intel UHD Graphics", "Intel Iris Xe Graphics", "NVIDIA GTX 1650", "NVIDIA RTX 3050", "NVIDIA RTX 3060", "NVIDIA RTX 3070", "AMD Radeon RX 6600M"]
ram_sizes = [4, 8, 16, 32, 64]
storage_types = ["256GB SSD", "512GB SSD", "1TB SSD", "1TB HDD + 256GB SSD"]

# Use cases with their RAM and CPU requirements
use_cases = {
    "coding": {"ram": (16, 64), "cpu": "high"},
    "testing": {"ram": (8, 16), "cpu": "medium"},
    "ML tasks": {"ram": (16, 64), "cpu": "high"},
    "gaming": {"ram": (32, 64), "cpu": "high"},
    "video editing": {"ram": (32, 64), "cpu": "high"},
    "office work": {"ram": (8, 16), "cpu": "medium"},
    "graphic design": {"ram": (8, 32), "cpu": "medium"},
    "everyday tasks": {"ram": (4, 8), "cpu": "low"}
}

# User types
user_types = ["coders", "testers", "ML engineers", "creatives", "students", "marketing", "gamers"]

# Review templates
review_templates = [
    "Great {performance} for {use_case}. The {ram} RAM and {cpu} make {task} a breeze.",
    "Excellent {feature}, perfect for {user_type}. {pros}, but {cons}.",
    "{pros}. Ideal for {use_case} with its {cpu}. {cons}.",
    "Budget-friendly option with {performance} specs. Good for {use_case}, but {cons}.",
]

# Laptop brand names
laptop_brands = ["TechPro", "InnoBook", "PowerLap", "SmartTech", "EliteBook"]

def generate_laptop_name(brand):
    """Generate a laptop name."""
    series = random.choice(["X", "Y", "Z", "Pro", "Air", "Ultra"])
    model = random.randint(1000, 9999)
    return f"{brand} {series}{model}"

def generate_specs(use_case):
    """Generate laptop specifications based on the given use case.

    Args:
        use_case (str): The intended use case for the laptop.
    Returns:
        dict: A dictionary containing the generated laptop specifications.
    """

    # Select a CPU based on the required performance tier for the use case
    cpu_tier = use_cases[use_case]["cpu"]
    cpu = random.choice(cpus[cpu_tier])
    gpu = random.choice(gpus)
    min_ram, max_ram = use_cases[use_case]["ram"]
    ram = random.choice([r for r in ram_sizes if min_ram <= r <= max_ram])
    storage = random.choice(storage_types)
    
    # Ensure ML tasks and gaming have a dedicated GPU
    if use_case in ["ML tasks", "gaming"]:
        gpu = random.choice([g for g in gpus if "NVIDIA" in g or "AMD Radeon" in g])
    
    # Generate a laptop name
    brand = random.choice(laptop_brands)
    name = generate_laptop_name(brand)
    
    return {
        "name": name,
        "ram": f"{ram}GB",
        "gpu": gpu,
        "cpu": cpu,
        "storage": storage
    }

def validate_specs(specs, use_case):
    """Validate the generated laptop specifications to ensure they meet the use case requirements.

    Args:
        specs (dict): The generated laptop specifications.
        use_case (str): The intended use case for the laptop.
    Returns:
        bool: True if the specifications meet the use case requirements, otherwise False.
    """

    # Extract RAM and CPU information from the specs
    ram = int(specs["ram"].split('GB')[0])
    cpu = specs["cpu"]

    # Get the required RAM and CPU tier for the use case
    min_ram, max_ram = use_cases[use_case]["ram"]
    cpu_tier = use_cases[use_case]["cpu"]

    if not (min_ram <= ram <= max_ram):
        return False
    if cpu_tier == "high" and not any(c in cpu for c in ["i7", "i9", "Ryzen 7", "Ryzen 9"]):
        return False
    
    # Ensure ML tasks and gaming have a dedicated GPU
    if use_case in ["ML tasks", "gaming"] and not any(g in specs["gpu"] for g in ["NVIDIA", "AMD Radeon"]):
        return False
    return True

def generate_review(specs, use_case):
    """Generate a review for the laptop based on its specifications and use case.

    Args:
        specs (dict): The generated laptop specifications.
        use_case (str): The intended use case for the laptop.
    Returns:
        str: A generated review for the laptop.
    """

    # Select a random template and populate it with relevant details
    template = random.choice(review_templates)
    performance = random.choice(["excellent", "good", "decent", "impressive"])
    task = random.choice(["multitasking", "rendering", "computing", "productivity"])
    feature = random.choice(["battery life", "display", "portability", "build quality"])
    user_type = random.choice(user_types)
    pros = random.choice(["Lightweight and portable", "High-resolution display", "Fast boot times", "Quiet operation"])
    cons = random.choice(["the price is on the higher side", "the processor is a bit slow for heavy tasks", "it struggles with demanding applications", "the build quality could be better"])

    review = template.format(
        performance=performance, use_case=use_case, ram=specs["ram"],
        gpu=specs["gpu"], cpu=specs["cpu"],
        task=task, feature=feature, user_type=user_type,
        pros=pros, cons=cons
    )
    return review

def generate_laptop_data(num_laptops):
    """Generate a specified number of laptops with specifications and reviews.

    Args:
        num_laptops (int): The number of laptops to generate.
    Returns:
        tuple: Two dictionaries containing laptop specifications and reviews.
               - specs (dict): Mapping of laptop IDs to their specifications.
               - reviews (dict): Mapping of laptop IDs to their reviews.
    """

    specs = {}
    reviews = {}
    use_case_count = {uc: 0 for uc in use_cases}

    while len(specs) < num_laptops:
        if len(specs) < len(use_cases):
            # Ensure each use case is represented at least once
            remaining_use_cases = [uc for uc, count in use_case_count.items() if count == 0]
            use_case = random.choice(remaining_use_cases)
        else:
            use_case = random.choice(list(use_cases.keys()))
        
        laptop_specs = generate_specs(use_case)
        
        if validate_specs(laptop_specs, use_case):
            laptop_id = f"laptop_{len(specs) + 1}"
            laptop_review = generate_review(laptop_specs, use_case)
            
            # Store the generated specs and review
            specs[laptop_id] = laptop_specs
            reviews[laptop_id] = laptop_review
            use_case_count[use_case] += 1
            
            logging.info(f"Generated {laptop_id} ({laptop_specs['name']}) for {use_case}: {laptop_specs}")
        else:
            logging.warning(f"Invalid specs generated for {use_case}: {laptop_specs}")

    logging.info(f"Use case distribution: {use_case_count}")
    return specs, reviews

def save_dict_to_json(dictionary, filename):
    """Save a dictionary to a file in JSON format.

    Args:
        dictionary (dict): The dictionary to save.
        filename (str): The path to the output file.
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=2)

# Generate data for a specified number of laptops
num_laptops = 20
specs, reviews = generate_laptop_data(num_laptops)

# Save the generated laptop specs and reviews to separate JSON files
save_dict_to_json(specs, "data/laptop_specs.json")
save_dict_to_json(reviews, "data/laptop_reviews.json")