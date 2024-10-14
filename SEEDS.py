import numpy as np
import cv2
from skimage.color import rgb2lab
from sklearn.metrics import pairwise

def compute_histogram(patch, num_bins=32):
    """Calcula un histograma de color normalizado para una región dada."""
    hist, _ = np.histogram(patch, bins=num_bins, range=(0, 256))
    hist = hist.astype(float)
    hist /= hist.sum()  # Normaliza el histograma
    return hist

def energy_function(pixel_color, hist, num_bins=32):
    """Calcula la energía basada en la diferencia de color."""
    # Calcula la distancia de color entre el píxel y el histograma
    bin_size = 256 // num_bins
    bin_index = pixel_color // bin_size
    color_dist = 1 - hist[bin_index]  # Invertimos la frecuencia
    return color_dist

def seeds_segmentation(image, num_superpixels, patch_size):
    """Segmentación por súper píxeles usando el algoritmo SEEDS."""
    height, width, _ = image.shape
    # Convertir la imagen a CIELAB
    lab_image = rgb2lab(image)
    
    # Inicializar los centroides aleatorios
    centroids = np.random.randint(0, min(height, width), size=(num_superpixels, 2))
    labels = -np.ones((height, width), dtype=int)

    for iteration in range(10):  # Número de iteraciones
        # Construir histogramas de color para cada súper píxel
        histograms = []
        for i in range(num_superpixels):
            x, y = centroids[i]
            patch = lab_image[max(0, y-patch_size//2):min(height, y+patch_size//2), 
                              max(0, x-patch_size//2):min(width, x+patch_size//2)]
            histogram = compute_histogram(patch)
            histograms.append(histogram)

        histograms = np.array(histograms)

        # Asignar píxeles a súper píxeles
        for y in range(height):
            for x in range(width):
                pixel_color = lab_image[y, x]
                min_energy = float('inf')
                best_label = -1

                for i in range(num_superpixels):
                    energy = energy_function(pixel_color, histograms[i])
                    spatial_dist = np.linalg.norm(centroids[i] - np.array([x, y]))
                    total_energy = energy + spatial_dist  # Combina energía de color y espacial

                    if total_energy < min_energy:
                        min_energy = total_energy
                        best_label = i

                labels[y, x] = best_label

        # Actualizar centroides
        for i in range(num_superpixels):
            # Obtener píxeles asignados a este súper píxel
            mask = (labels == i)
            if np.any(mask):
                y_indices, x_indices = np.where(mask)
                new_centroid = np.mean(np.array([x_indices, y_indices]), axis=1)
                centroids[i] = new_centroid.astype(int)

    return labels

# Uso del algoritmo
if __name__ == "__main__":
    # Cargar una imagen
    image = cv2.imread('./peabody.jpg')  # Asegúrate de poner la ruta correcta
    num_superpixels = 100  # Número de súper píxeles deseados
    patch_size = 20  # Tamaño del parche

    labels = seeds_segmentation(image, num_superpixels, patch_size)

    # Visualización de los resultados
    output_image = np.zeros_like(image)
    for label in range(num_superpixels):
        output_image[labels == label] = image[labels == label].mean(axis=0)

    cv2.imshow('Segmented Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
