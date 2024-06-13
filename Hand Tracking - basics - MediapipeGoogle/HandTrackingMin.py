import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Creer le modele de detection des mains
mpHands = mp.solutions.hands
# creer un objet k'on appelera hands
hands = mpHands.Hands()

#pour marque(dessiner) le 21 point  de la main on a fonction:
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    
    #convertir notre image en RGB car notre modele utilise uniquement des images RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #pour voir si ma main est detecter ou pas
    #print(results.multi_hand_landmarks)

    #Donc une boucle pour dire que chaque main repere on recupere le point de la main"21point" 
    #Puis en etabli la connection des points avec HAND_CONNECTIONS
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()