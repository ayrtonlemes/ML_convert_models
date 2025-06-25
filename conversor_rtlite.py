import ai_edge_torch
import torch
import timm

# Cria o modelo exatamente como foi treinado
#Aqui seleciona o modelo no formato do arquivo de geração, em seguida o modelo gerado, e o nome do modelo de saida

model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=3) #Modelo base do timm
model.load_state_dict(torch.load("/home/ayrton/Área de Trabalho/LEGO_ML/convertendo_modelos/modelos_gerados/swin_tiny_best.pth", map_location="cpu")) #Modelo pytorch pra carregar os state_dict
model.eval()

# Input de exemplo
sample_inputs = (torch.randn(1, 3, 224, 224),)

# Converte para TFLite
edge_model = ai_edge_torch.convert(model, sample_inputs)
edge_model.export("swin_tiny_best.tflite")
print("Modelo exportado com sucesso!")
