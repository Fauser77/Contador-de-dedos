import mediapipe as mp
import cv2

video = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=2)

# Responsável pelos desenhos dos pontos
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = Hand.process(imgRGB)

    handsPoints = results.multi_hand_landmarks

    # Extraindo informações da mão
    # (label para "Right" ou "Left" para identificar qual mão é)
    handedness = results.multi_handedness

    h, w, _ = img.shape
    total_dedos = 0

    if handsPoints:
        for hand_idx, points in enumerate(handsPoints):
            lado = handedness[hand_idx].classification[0].label
           
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)

            pontos = []
            for id, cord in enumerate(points.landmark):
                # Convertendo cada ponto em pixels
                cx, cy = int(cord.x * w), int(cord.y * h)

                # Desenho de um círculo em cada ponto
                cv2.putText(img, str(id), (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                pontos.append((cx, cy))

            # Marcação de dedos por pontos
            dedos = [8, 12, 16, 20]
            contador = 0

            if pontos:
                # Processamento para o polegar
                if (lado == 'Right' and pontos[4][0] < pontos[2][0]) \
                    or (lado == 'Left' and pontos[4][0] > pontos[2][0]):
                        contador += 1

                # Processamento para outros dedos 
                for x in dedos:
                    if pontos[x][1] < pontos[x - 2][1]:
                        contador += 1   
            
                # Exibição do total de dedos por mão
                if lado == 'Right':
                    cv2.rectangle(img,
                                  (515, 30),
                                  (320, 100),
                                  (0, 255, 0),
                                  -1)
                    cv2.putText(img,
                                f'Esquerda: {contador}',
                                (320, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 0),
                                2)
                else:
                    cv2.rectangle(img,
                                  (100, 30),
                                  (250, 100),
                                  (0, 255, 0),
                                  -1)
                    cv2.putText(img,
                                f'Direita: {contador}',
                                (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 0),
                                2)

            total_dedos += contador

    # Exibição do total de dedos das duas mãos
    cv2.rectangle(img,
                  (10, 420),
                  (320, 460),
                  (0, 255, 0),
                  -1)
    cv2.putText(img,
                f'Total de dedos: {total_dedos}',
                (15, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0), 2)

    cv2.imshow("Imagem", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

