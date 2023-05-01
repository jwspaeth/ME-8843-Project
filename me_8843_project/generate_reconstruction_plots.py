import cv2
import numpy as np
from PIL import Image


def fetch_frame(video_path, frame_percent):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Get total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Read until video is completed
    count = 0
    while cap.isOpened() and count < total_frames * frame_percent:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow("Frame", frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Break the loop
        else:
            break

        count += 1

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return frame


def generate_reconstruction_plots():

    # Fetch lunar lander data
    lunar_frame = fetch_frame("lunar_vid/recon_episode_4.mp4", 0.5)

    # Fetch hopper data
    hopper_frame = fetch_frame("hopper_vid/recon_episode_4.mp4", 0.5)

    # Fetch reacher data
    reacher_frame = fetch_frame("reacher_vid/recon_episode_4.mp4", 0.25)

    # Fetch swimmer data
    swimmer_frame = fetch_frame("swimmer_vid/recon_episode_4.mp4", 0.5)

    # Join frame vertically
    top = np.concatenate((lunar_frame, hopper_frame), axis=1)
    bottom = np.concatenate((reacher_frame, swimmer_frame), axis=1)
    join = np.concatenate((top, bottom), axis=0)

    # Display the resulting frame
    cv2.imshow("Frame", join)
    cv2.waitKey(0)

    # Save the resulting frame
    im = Image.fromarray(join)
    im.save("figures/reconstruction_plot.png")


if __name__ == "__main__":
    generate_reconstruction_plots()
