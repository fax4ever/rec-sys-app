import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def load_clip_model():
    """Load the CLIP model and processor."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def get_text_embedding(text, model, processor):
    """
    Generate CLIP embedding for a text input.
    
    Args:
        text (str): Input text
        model: CLIP model
        processor: CLIP processor
        
    Returns:
        numpy.ndarray: Text embedding
    """
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.numpy()

def get_image_embedding(image_path, model, processor):
    """
    Generate CLIP embedding for an image.
    
    Args:
        image_path (str): Path to the image file
        model: CLIP model
        processor: CLIP processor
        
    Returns:
        numpy.ndarray: Image embedding
    """
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.numpy()

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1 (numpy.ndarray): First embedding
        embedding2 (numpy.ndarray): Second embedding
        
    Returns:
        float: Cosine similarity score
    """
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1, axis=1, keepdims=True)
    embedding2 = embedding2 / np.linalg.norm(embedding2, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarity = np.dot(embedding1, embedding2.T)
    return similarity[0][0]

def main():
    # Load model and processor
    model, processor = load_clip_model()
    
    # Example 1: Text embedding
    text = "a photo of a cat sitting on a windowsill"
    text_embedding = get_text_embedding(text, model, processor)
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Example 2: Image embedding (replace with your image path)
    # image_path = "path/to/your/image.jpg"
    # image_embedding = get_image_embedding(image_path, model, processor)
    # print(f"Image embedding shape: {image_embedding.shape}")
    
    # Example 3: Multiple text embeddings and similarity
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird"
    ]
    
    # Generate embeddings for all texts
    text_embeddings = []
    for text in texts:
        embedding = get_text_embedding(text, model, processor)
        text_embeddings.append(embedding)
    
    # Compute similarities between the first text and others
    print("\nSimilarity scores:")
    for i, text in enumerate(texts[1:], 1):
        similarity = compute_similarity(text_embeddings[0], text_embeddings[i])
        print(f"Similarity between '{texts[0]}' and '{text}': {similarity:.4f}")

if __name__ == "__main__":
    main() 