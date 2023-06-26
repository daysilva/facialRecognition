#!/usr/bin/python3
import cv2
import threading
import time
import tempfile

from recognition.recognition import Recognition

# declarar variaveis
PROCESSO_ATIVO = False
HORA_DETECCAO = 0
CADASTRAR_ROSTO = True
ARQUIVO_TEMPORARIO = None


# instanciar a classe com o metodo de reconhecimento
# esta classe contem o atributo q me retornara o resultado
recog = Recognition()

haar_cascade_xml = 'haarcascade_frontalface_alt2.xml'

faceClassifier = cv2.CascadeClassifier(haar_cascade_xml)

# # iniciar a camera
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while not cv2.waitKey(20) & 0xFF == ord('q'):
    try:

        time.sleep(0.5)
        HORA_DETECCAO += 1

        ret, frame = capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face = faceClassifier.detectMultiScale(gray)

        #criar retangulo que detecta rosto
        for x, y, w, h in face:
            face_capture = cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

            # detectar um rosto, chamar a função de fazer reconhecimento
            if len(face_capture) > 0:
                rosto = frame[y:y+h, x:x+w]

                # cv2.imwrite("./fotos_teste/ft.jpg" , rosto)

                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    ARQUIVO_TEMPORARIO = temp_file.name

                    # Salva a imagem do rosto no arquivo temporário
                    cv2.imwrite(ARQUIVO_TEMPORARIO, rosto)
            
            
        if PROCESSO_ATIVO == False:
            print('vai')
            PROCESSO_ATIVO = True
            thread = threading.Thread(target=recog.recognition, args=['./clau.jpeg', ARQUIVO_TEMPORARIO]).start()
    
            
        if recog.result != None:
            if recog.result != 10 and CADASTRAR_ROSTO == True:
                
                print(">>", recog.result)
                CADASTRAR_ROSTO = False # so cadastraremos um novo rosto daqui a 10 min por ex.

                # chamar a funcao que salvará no banco
                # salvarNotificacao("fulano entrou no carro")

                recog.result = None # lipar esse resultado para um novo

            
            else:
                # caso o resultado seja 10 significa que o metodo
                # nao conseguiu fazer o reconhecimento
                # entao queremos executar o metodo mais vezes
                PROCESSO_ATIVO = False


        # virificar se ja passaram os 10 min para poder
        # fazer uma nova identificação
        if HORA_DETECCAO > 10:
            PROCESSO_ATIVO = False
            CADASTRAR_ROSTO = True
            HORA_DETECCAO = 0

        # cv2.imshow('gray', gray)
        cv2.imshow('color', frame)
    except:
        print("erro")

capture.release()
cv2.destroyAllWindows()