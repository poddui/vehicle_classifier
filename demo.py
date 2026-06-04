import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Käytetään näytönohjainta prosessorin sijaan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kuvien transformointi
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225 ])
])

# Ladataan datasetit
train_dataset = datasets.ImageFolder("data/train", transform=transform)
val_dataset = datasets.ImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class_names = train_dataset.classes
print(f"Luokat: {class_names}")

# Ladataan esikoulutettu malli
model = models.resnet18(weights='DEFAULT')

# Jäädytetään kaikki kerrokset
for param in model.parameters():
    param.requires_grad = False

# Korvataan viimeinen kerros
num_classes = 4 
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Jos tallennettu malli löytyy käytetään sitä, muuten luodaan uusi
model_path = "vehicle_model.pth"
if os.path.exists(model_path):
    print("Tallennettu malli löytyi, ladataan painot...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
else:
    print("Mallia ei löydy, aloitetaan koulutus alusta...")

# Siirretään laskenta näytönohjaimelle
model = model.to(device)

# Aktivoidaan ristientropia
criterion = nn.CrossEntropyLoss()

# Aktivoidaan optimoija Adam
optimizer = optim.Adam(model.fc.parameters(), lr=0.01)

# Harjoitussilmukka
num_epochs = 15
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Häviö: {running_loss/len(train_loader):.4f}")

# Tallennetaan mallin painot
torch.save(model.state_dict(), "vehicle_model.pth")
print("Mallin painot tallennettu!")

# Mallin testaus uusilla kuvilla
model.eval()
correct = 0
total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = correct / total * 100
print(f'Tarkkuus validointisetillä: {accuracy:.2f}%')

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix - Ajoneuvoluokittelu")
plt.ylabel("Oikea luokka")
plt.xlabel("Ennustettu luokka")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi= 150)
plt.show()
print("Confusion matrix tallennettu!")

# Häviökuvaaja
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker="o", color="steelblue")
plt.title('Koulutushäviö epocheittain')
plt.xlabel('Epoch')
plt.ylabel('Häviö (Loss)')
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()
print("Häviökuvaaja tallennettu!")



def show_predictions(loader, model, class_names, num_images=16):
    model.eval()
    per_class = {name: {"oikein": [], "vaarin": []} for name in class_names}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(images.size(0)):
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                # Perutaan normalisointi
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                oikea = class_names[labels[i]]
                ennuste = class_names[predicted[i]]

                if oikea == ennuste and len(per_class[oikea]["oikein"]) < 3:
                    per_class[oikea]["oikein"].append((img, oikea, ennuste))
                elif oikea != ennuste and len(per_class[oikea]["vaarin"]) < 1:
                    per_class[oikea]["vaarin"].append((img, oikea, ennuste))

            if all(len(v["oikein"]) == 3 and len(v["vaarin"]) == 1 for v in per_class.values()):
                break

    fig, axes = plt.subplots(4, 4, figsize=(14, 7))
    axes = axes.flatten()
    idx = 0
    for slot in range(4):  # 0=3 oikeaa, 3=1 väärä
        for luokka in class_names:
            if slot < 3:
                samples = per_class[luokka]["oikein"]
                if slot < len(samples):
                    img, oikea, ennuste = samples[slot]
                    axes[idx].imshow(img)
                    axes[idx].axis('off')
                    axes[idx].set_title(f'Oikea: {oikea}\nEnnuste: {ennuste}', color='green', fontsize=9)
            else:
                samples = per_class[luokka]["vaarin"]
                if len(samples) > 0:
                    img, oikea, ennuste = samples[0]
                    axes[idx].imshow(img)
                    axes[idx].axis('off')
                    axes[idx].set_title(f'Oikea: {oikea}\nEnnuste: {ennuste}', color='red', fontsize=9)
            idx += 1

    plt.suptitle('Per luokka: 3 oikein (vihreä) + 1 väärin (punainen)', fontsize=12)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()
    print("Ennustekuvat tallennettu!")

show_predictions(val_loader, model, class_names)




