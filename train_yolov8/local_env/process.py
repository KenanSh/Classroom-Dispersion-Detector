import os
import cv2

# Directories containing the text files and images
label_directory = '.\train_yolov8\data\labels\val'
image_directory = '.\train_yolov8\data\images\val'

# Define class mapping
class_mapping = {
    'Person': 0,
    'Human face': 1,
    'person': 0,
    'human face': 1
}

# Iterate over each file in the directory
for filename in os.listdir(label_directory):
    # Process only .txt files
    if filename.endswith('.txt'):  
        label_filepath = os.path.join(label_directory, filename)
        image_filename = filename.replace('.txt', '.jpg')
        image_filepath = os.path.join(image_directory, image_filename)

        # Check if the corresponding image file exists
        if not os.path.exists(image_filepath):
            print(f"Image file {image_filename} not found for {filename}. Skipping...")
            continue

        print(f"Processing {filename}...")

        # Read image to get its dimensions
        image = cv2.imread(image_filepath)
        if image is None:
            print(f"Could not read image {image_filename}. Skipping...")
            continue

        img_height, img_width = image.shape[:2]

        # Read contents of the label file
        with open(label_filepath, 'r') as file:
            lines = file.readlines()

        # Modify lines to YOLO format: replace class names with class indices and convert bounding boxes
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Skipping invalid line in {filename}: {line.strip()}")
                continue  # Skip lines that do not have enough parts (invalid format)

            class_name = ' '.join(parts[:-4]).strip()
            if class_name in class_mapping:
                class_index = class_mapping[class_name]
                try:
                    x_min = float(parts[-4])
                    y_min = float(parts[-3])
                    x_max = float(parts[-2])
                    y_max = float(parts[-1])
                except ValueError:
                    print(f"Skipping line with invalid coordinates in {filename}: {line.strip()}")
                    continue  # Skip lines where coordinates are not valid numbers

                # Convert bounding box coordinates to normalized values
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # Format the converted line in YOLO format
                converted_line = f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                converted_lines.append(converted_line)
            else:
                print(f"Unknown class name '{class_name}' in file {filename}")

        # Write converted content back to the file
        with open(label_filepath, 'w') as file:
            file.writelines(converted_lines)

        print(f"{filename} processed successfully.")

print("All files processed.")
