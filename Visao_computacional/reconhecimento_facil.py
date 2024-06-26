import cv2

# Carregamento da imagem
imagem = cv2.imread('../Dados/workplace-1245776_1920.jpg')

# Carregamento do classificador para detecção facial
detector_facial = cv2.CascadeClassifier('../Dados/haarcascade_frontalface_default.xml')

# Conversão da imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detecção de faces na imagem
deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.3)

# Desenha retângulos ao redor das faces detectadas
for (x, y, l, a) in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

# Verifica se a imagem foi carregada com sucesso
if imagem is not None:
    # Obtém as dimensões originais da imagem
    height, width = imagem.shape[:2]

    # Define o novo tamanho desejado da imagem (por exemplo, 1280x800)
    nova_largura = 1280
    nova_altura = 800

    # Redimensiona a imagem para o novo tamanho
    imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura))

    # Exibe a imagem com as detecções de faces
    cv2.imshow('Detecção de Faces', imagem_redimensionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Imagem não carregada com sucesso.')
