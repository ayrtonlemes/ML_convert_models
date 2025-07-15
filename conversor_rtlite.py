import ai_edge_torch
import torch
from torchvision import models

#from ai_edge_torch.convert.converter import convert
# === Define a quantidade de classes
num_classes = 3

# === Cria o modelo ResNet18 com a última camada ajustada
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# === Carrega os pesos do modelo treinado
model.load_state_dict(torch.load("resnet18_classificador_emocoes.pth", map_location="cpu"))
model.eval()

# === Define input de exemplo (mesma forma usada no treino/inferência)
sample_inputs = (torch.randn(1, 3, 224, 224),)

# === Converte para TFLite com ai_edge_torch
edge_model = ai_edge_torch.convert(model.eval(), sample_inputs) #usar o converter direto

edge_model.export("resnet18_classificador_emocoes.tflite")

print("✅ Modelo TFLite exportado com sucesso!")

