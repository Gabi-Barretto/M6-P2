import cv2
from time import sleep

# Carrega os classificadores em cascata
face_cascade = cv2.CascadeClassifier(
    filename=f"{cv2.data.haarcascades}/haarcascade_frontalface_default.xml"
)

# Abre o arquivo de video
input_video = cv2.VideoCapture('./arsene.mp4')

# Checa se foi possivel abrir o arquivo
if not input_video.isOpened():
    print("Error opening video file")
    exit(1)
    
# Como foi possível abrir o video de entrada, vamos agora utilizar 
# essa captura para definir o tamanho do video de saida
width  = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Cria a estrutura do video de saida
# Com formato e local do arquivo de saida
# Codec utilizado
# FPS do video e
# Tamanho do video
output_video = cv2.VideoWriter( './out2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 24, (width, height))

# Loop de leitura frame por frame
while True:
    # Le um frame do video e, guarda o resultado da leitura
    # Se nao houver mais frames disponiveis, ret sera falso
    ret, frame = input_video.read()

    gray_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        image=gray_frame, 
        scaleFactor=1.15, # Mudança de escala a cada passada
        minNeighbors=25 # Verifica os vizinhos antes de promover o ponto a ret
    )

    # Se nao conseguiu ler o frame, para o laco
    if not ret:
        break

    if len(faces) > 0:
        x, y, w, h = faces[0]
    
        # Vamos editar o frame com um retangulo
        cv2.rectangle(
                img=frame,
                pt1=(x, y),
                pt2=(x+w, y+h),
                color=(0,0,255),
                thickness=3
            )

    # Exibe o frame
    cv2.imshow('Video Playback', frame)
    
    # Escreve o frame no output
    output_video.write(frame)

    # Se o usuario apertar q, encerra o playback
    # O valor utilizado no waiKey define o fps do playback
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
# Fecha tudo
output_video.release()
input_video.release()
cv2.destroyAllWindows()
