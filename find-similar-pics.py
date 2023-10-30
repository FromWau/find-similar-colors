#!/usr/bin/env python3

from PIL import Image
import os
from sklearn.cluster import KMeans
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse


# Extract and sort the dominant colors
def extract_color_scheme(image_path, n_colors=5):
    image = Image.open(image_path)
    image = image.convert('RGB')

    image = image.resize((100, 100))

    image_data = np.array(image)

    pixels = image_data.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)

    color_counts = np.bincount(kmeans.labels_)
    sorted_colors = [colors[i] for i in np.argsort(color_counts)[::-1]]

    return sorted_colors

def compare_color_schemes(color_scheme1, color_scheme2):
    color_scheme1_mean = np.mean(color_scheme1, axis=0)
    color_scheme2_mean = np.mean(color_scheme2, axis=0)
    return np.linalg.norm(color_scheme1_mean - color_scheme2_mean)

def find_matching_images(image, dir, num_matches=2):
    input_color_scheme = extract_color_scheme(image)

    matching_images = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for filename in os.listdir(dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_path = os.path.join(dir, filename)
                futures.append(executor.submit(extract_color_scheme, image_path))

        for future, image_path in zip(futures, os.listdir(dir)):
            color_scheme = future.result()
            similarity = compare_color_schemes(input_color_scheme, color_scheme)
            matching_images.append((image_path, similarity))

    matching_images.sort(key=lambda x: x[1])
    matching_images = matching_images[:num_matches]

    return matching_images

def main():
    parser = argparse.ArgumentParser(description="Find similar images based on color schemes.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("dir", help="Path to the directory containing images to compare.")
    parser.add_argument("-n", "--num-matches", type=int, default=2, help="Number of similar images to find.")
    parser.add_argument("-r", "--raw", action="store_true", help="Display only the raw image paths.")
    
    args = parser.parse_args()

    matches = find_matching_images(args.image, args.dir, args.num_matches)

    for i, (image_path, similarity) in enumerate(matches):
        abs_image_path = os.path.abspath(os.path.join(args.dir, image_path))
        if args.raw:
            print(abs_image_path)
        else:
            abs_image_path = os.path.abspath(os.path.join(args.dir, image_path))
            print(f"Match {i + 1} - Similarity: {similarity:.2f}")
            print(f"Image Path: {abs_image_path}")

if __name__ == "__main__":
    main()
