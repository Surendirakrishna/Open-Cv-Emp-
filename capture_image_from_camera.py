import cv2
from matplotlib import pyplot as plt

cam_port = 0
cam = cv2.VideoCapture(cam_port)

# Reading the input using the camera
inp = input('Enter person name: ')

while True:
    result, image = cam.read()
    if result:
        # Display the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(inp)
        plt.show()

        # Ask the user if they want to save the image
        save = input("Do you want to save the image? (y/n): ")
        if save.lower() == 'y':
            cv2.imwrite(inp + ".png", image)
            print("Image saved as", inp + ".png")
            break
        else:
            print("Image not saved. Retaking image.")
    else:
        print("No image detected. Please try again.")

# Release the camera
cam.release()
