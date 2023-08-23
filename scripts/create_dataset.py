import os
import glob
import cv2
from uuid import uuid4
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

video_paths = [
    "videos/garden_of_words.mp4",
    "videos/weathering_with_you.mp4",
    "videos/your_name.mkv",
    "videos/suzume.mp4"
]

directories = [
    "images/Target/garden_of_words",
    "images/Target/weathering_with_you",
    "images/Target/your_name",
    "images/Target/suzume"
]

def extract_frames():
    for video_path in video_paths:
        name = video_path.split("/")[-1].split(".")[0]
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()

        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        for scene in scene_list:
            start_frame = int(scene[0].get_frames())
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()

            if ret:
                # Save the frame as an image
                frame_filename = f"images/Target/{name}/" + str(uuid4()) + f"{start_frame}.jpg"
                cv2.imwrite(frame_filename, frame)

        cap.release()

def clean_images():
    image_extensions = ['jpg', 'jpeg', 'png']
    all_image_paths = []

    for directory in directories:
        for extension in image_extensions:
            pattern = os.path.join(directory, f'*.{extension}')
            image_paths = glob.glob(pattern)
            all_image_paths.extend(image_paths)

    brightness_threshold = 25

    dark_images = []

    for image_path in all_image_paths:
        
        # Load the image
        img = cv2.imread(image_path)
        
        # Check if image is loaded successfully
        if img is not None:
            # Convert image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate average pixel value (brightness)
            brightness = cv2.mean(gray_img)[0]
            
            if brightness < brightness_threshold:
                dark_images.append(image_path)
        else:
            print(f"Failed to load image: {image_path.split('/')[-1]}")
    
    for img in dark_images:
        os.remove(img)

if __name__ == "__main__":
    extract_frames()
    print("Image Extraction Ended!!!")
    clean_images()
    print("Image Cleaning Ended!!!")


