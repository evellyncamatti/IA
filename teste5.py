import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def load_image(file_path):
    # Carrega a imagem
    img = cv2.imread(file_path)
    # Converte a imagem para RGB (OpenCV lê em BGR por padrão)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_images(images, titles):
    # Plota imagens com títulos
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    
    # Se houver apenas um subplot, axes não precisa ser iterado
    if len(images) == 1:
        axes.imshow(images[0])
        axes.set_title(titles[0])
        axes.axis('off')
    else:
        for img, title, ax in zip(images, titles, axes):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
    
    plt.show()

def compute_image_properties(img, file_path=None):
    # Computa propriedades da imagem (pode adicionar mais conforme necessário)
    properties = {
        'Resolução': img.shape,
        'Tamanho (bytes)': os.path.getsize(file_path) if file_path else img.nbytes,
        'Média': np.mean(img),
        'Desvio Padrão': np.std(img),
        'Cores Únicas': len(np.unique(img.reshape(-1, img.shape[2]), axis=0)),
    }
    return properties

def apply_kmeans(img, k):
    # Redimensiona a imagem para um array 2D
    pixels = img.reshape((-1, 3))
    
    # Aplica o algoritmo k-médias
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Substitui cada pixel pelo valor do centróide do cluster correspondente
    segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(img.shape).astype(np.uint8)

    return segmented_img

def main():
    # Caminho da imagem
    file_path = 'teste.png'
    
    # Carrega a imagem original
    original_img = load_image(file_path)

    # Computa propriedades da imagem original
    original_properties = compute_image_properties(original_img, file_path)

    # Exibe a imagem original
    plot_images([original_img], ['Imagem Original'])

    # Escolhe o valor de k (número de clusters)
    k = 21

    # Computa propriedades da imagem antes de aplicar o algoritmo k-médias
    original_properties_before_kmeans = compute_image_properties(original_img, file_path)

    # Aplica o algoritmo k-médias
    segmented_img = apply_kmeans(original_img, k)

    # Salva a imagem segmentada com o nome "final.png"
    cv2.imwrite("final.png", cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

    # Computa propriedades da imagem após o uso do algoritmo k-médias
    segmented_properties_after_kmeans = compute_image_properties(segmented_img, "final.png")

    # Exibe a imagem segmentada
    plot_images([segmented_img], ['Imagem Segmentada'])

    # Exibe as propriedades antes e depois de aplicar o k-médias
    print("\nPropriedades da Imagem Original:")
    for key, value in original_properties_before_kmeans.items():
        print(f"{key}: {value}")

    print("\nPropriedades da Imagem Segmentada Após K-Médias:")
    for key, value in segmented_properties_after_kmeans.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
