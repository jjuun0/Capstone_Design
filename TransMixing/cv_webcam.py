import cv2

def setting():
     #0이면 노트북 내장 웹캠 숫자를 올리면 추가된 웹캠을 이용할 수 있다.
    cap = cv2.VideoCapture(0) # 3은 가로 4는 세로 길이
    # cap.set(3, 720)
    # cap.set(4, 1080)
    while True:
        ret, frame = cap.read()
        cv2.imshow('test', frame)  # ndarray, shape=(h, w, c)
        k = cv2.waitKey(1)  # esc
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def capture(n):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = n+1

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # setting()

    capture(23)
