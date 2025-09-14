import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import gradio as gr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. Data preparation
# -----------------------------
data_dir = os.path.join(os.path.dirname(__file__), "cats_dogs")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("Loading dataset from:", data_dir)
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = full_dataset.classes
print("Found classes:", class_names)
print("Total images:", len(full_dataset))

# -----------------------------
# 2a. Quick test: small subset
# -----------------------------
small_dataset_size = 1000
if small_dataset_size < len(full_dataset):
    indices = random.sample(range(len(full_dataset)), small_dataset_size)
    dataset = torch.utils.data.Subset(full_dataset, indices)
else:
    dataset = full_dataset

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# -----------------------------
# 3. Model loader with save/load functionality
# -----------------------------
def get_model(name="resnet18", num_classes=2, freeze_backbone=True):
    print(f"Loading pretrained {name} weights (if not cached)...")
    if name == "resnet18":
        model = models.resnet18(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(pretrained=True)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Choose 'resnet18' or 'vgg16'")
    print("Model ready!")
    return model.to(device)

def load_trained_model(model_path, model_name="resnet18", num_classes=2):
    """Load a pre-trained model if it exists"""
    if os.path.exists(model_path):
        print(f"Loading existing trained model from: {model_path}")
        model = get_model(model_name, num_classes, freeze_backbone=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Pre-trained model loaded successfully!")
        return model, True
    else:
        print(f"No existing model found at: {model_path}")
        print("Will train a new model...")
        return None, False

# -----------------------------
# 4. Training function with accuracy metrics
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=2, lr=0.001, save_name="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print("\nTraining started...")
    print("-" * 50)
    
    best_val_acc = 0.0
    training_history = {'train_acc': [], 'val_acc': [], 'train_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        all_predictions, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        
        # Calculate detailed metrics
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Store history
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['train_loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"  New best model saved! (Val Acc: {val_acc:.2f}%)")
        print("-" * 50)

    # Generate final confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nFinal Confusion Matrix:")
    print("Predicted ->")
    print(f"Actual    {class_names[0]:<8} {class_names[1]:<8}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<8}  {cm[i][0]:<8} {cm[i][1]:<8}")
    
    print(f"\nModel saved as: {save_name}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return training_history

# -----------------------------
# 5. Enhanced prediction functions
# -----------------------------
def predict_single_image(image_input, model):
    """
    Predict class for a single image with confidence scores
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        confidence = probabilities[0][predicted].item()
    
    predicted_class = class_names[predicted.item()]
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': {class_names[i]: float(probabilities[0][i]) for i in range(len(class_names))}
    }

def predict_multiple_images(images_list):
    """
    Predict classes for multiple images and return batch results
    """
    results = []
    correct_predictions = 0
    
    for i, image_input in enumerate(images_list):
        try:
            result = predict_single_image(image_input, resnet_model)
            result['image_index'] = i + 1
            results.append(result)
        except Exception as e:
            results.append({
                'image_index': i + 1,
                'error': str(e),
                'prediction': 'Error',
                'confidence': 0.0
            })
    
    # Format results for display
    summary = f"Batch Prediction Results ({len(results)} images):\n"
    summary += "=" * 50 + "\n"
    
    for result in results:
        if 'error' not in result:
            summary += f"Image {result['image_index']}: {result['prediction']} "
            summary += f"(Confidence: {result['confidence']:.1%})\n"
            
            # Show probability breakdown
            probs = result['probabilities']
            summary += f"  Probabilities: "
            for class_name, prob in probs.items():
                summary += f"{class_name}: {prob:.1%}  "
            summary += "\n"
        else:
            summary += f"Image {result['image_index']}: Error - {result['error']}\n"
        summary += "-" * 30 + "\n"
    
    return summary

def gradio_single_predict(image):
    """Gradio interface for single image prediction"""
    if image is None:
        return "Please upload an image."
    
    result = predict_single_image(image, resnet_model)
    
    output = f"Prediction: {result['prediction']}\n"
    output += f"Confidence: {result['confidence']:.1%}\n\n"
    output += "Class Probabilities:\n"
    for class_name, prob in result['probabilities'].items():
        output += f"  {class_name}: {prob:.1%}\n"
    
    return output

def gradio_multiple_predict(files):
    """Gradio interface for multiple image prediction"""
    if not files or len(files) == 0:
        return "Please upload at least one image."
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Open image from file path
            image = Image.open(file.name).convert("RGB")
            result = predict_single_image(image, resnet_model)
            result['image_index'] = i + 1
            result['filename'] = os.path.basename(file.name)
            results.append(result)
        except Exception as e:
            results.append({
                'image_index': i + 1,
                'filename': os.path.basename(file.name) if hasattr(file, 'name') else f'Image_{i+1}',
                'error': str(e),
                'prediction': 'Error',
                'confidence': 0.0
            })
    
    # Format results for display
    summary = f"Batch Prediction Results ({len(results)} images):\n"
    summary += "=" * 60 + "\n"
    
    cat_count = 0
    dog_count = 0
    error_count = 0
    
    for result in results:
        if 'error' not in result:
            summary += f"Image {result['image_index']} ({result['filename']}):\n"
            summary += f"  Prediction: {result['prediction']} (Confidence: {result['confidence']:.1%})\n"
            
            # Count predictions
            if result['prediction'].lower() == 'cat':
                cat_count += 1
            elif result['prediction'].lower() == 'dog':
                dog_count += 1
            
            # Show probability breakdown
            probs = result['probabilities']
            summary += f"  Probabilities: "
            for class_name, prob in probs.items():
                summary += f"{class_name}: {prob:.1%}  "
            summary += "\n"
        else:
            summary += f"Image {result['image_index']} ({result['filename']}): Error - {result['error']}\n"
            error_count += 1
        summary += "-" * 40 + "\n"
    
    # Add summary statistics
    summary += f"\nSUMMARY STATISTICS:\n"
    summary += f"Total Images: {len(results)}\n"
    summary += f"Cats Detected: {cat_count}\n"
    summary += f"Dogs Detected: {dog_count}\n"
    summary += f"Errors: {error_count}\n"
    summary += f"Success Rate: {((len(results) - error_count) / len(results) * 100):.1f}%\n"
    
    return summary

# -----------------------------
# 6. Model evaluation function
# -----------------------------
def evaluate_model_accuracy(model, data_loader):
    """Evaluate model on validation set and return detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print("\nModel Evaluation Results:")
    print("=" * 40)
    print(f"Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# -----------------------------
# 7. Main execution
# -----------------------------
if __name__ == "__main__":
    MODEL_PATH = "quick_test_resnet18.pth"
    
    # Try to load existing model first
    resnet_model, model_loaded = load_trained_model(MODEL_PATH, "resnet18", len(class_names))
    
    if not model_loaded:
        # Only train if no existing model found
        print("\nTraining new model...")
        resnet_model = get_model("resnet18", freeze_backbone=True)
        training_history = train_model(resnet_model, train_loader, val_loader, epochs=3, save_name=MODEL_PATH)
        
        # Evaluate newly trained model
        print("\nEvaluating trained model...")
        model_metrics = evaluate_model_accuracy(resnet_model, val_loader)
    else:
        print("Using existing trained model. Skipping training.")
        print("If you want to retrain, delete the model file:", MODEL_PATH)

    # Create enhanced Gradio interfaces
    print("\nLaunching Gradio web application...")
    print("=" * 50)
    
    # Single image interface
    single_interface = gr.Interface(
        fn=gradio_single_predict,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Textbox(label="Prediction Results", lines=8),
        title="Cat vs Dog Classifier - Single Image",
        description="Upload a single image to classify as cat or dog with confidence scores and probability breakdown.",
        examples=None
    )
    
    # Multiple images interface  
    multiple_interface = gr.Interface(
        fn=gradio_multiple_predict,
        inputs=gr.File(file_count="multiple", file_types=["image"], label="Upload Multiple Images"),
        outputs=gr.Textbox(label="Batch Prediction Results", lines=20),
        title="Cat vs Dog Classifier - Multiple Images",
        description="Upload multiple images at once to classify them with detailed statistics and summary metrics.",
        examples=None
    )
    
    # Launch both interfaces in tabs
    demo = gr.TabbedInterface(
        [single_interface, multiple_interface],
        ["Single Image", "Batch Processing"],
        title="Professional Cat vs Dog Classifier",
        theme=gr.themes.Soft()
    )
    
    print("Starting web interface...")
    print("- Single Image Tab: Upload individual images for classification")
    print("- Batch Processing Tab: Upload multiple images for bulk analysis")
    print("- The interface will open in your browser automatically")
    
    demo.launch(share=True, show_error=True)
