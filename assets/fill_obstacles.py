import cv2
import numpy as np

# 1. Load the image
image_path = 'cave.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not find {image_path}")
    exit()

# 2. Invert the image
# OpenCV finds contours of WHITE objects on BLACK backgrounds.
# Currently, your map is Black walls on White background.
# So we invert it: Now obstacles are White Rings on Black.
inverted_img = cv2.bitwise_not(img)

# 3. Find Contours
# RETR_EXTERNAL ensures we only get the outer shell of the donut, ignoring the inner hole.
contours, _ = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} obstacles.")

# 4. Fill the Contours
# Draw filled contours (thickness=-1) on the inverted image
# This turns "White Rings" into "Solid White Blobs"
cv2.drawContours(inverted_img, contours, -1, (255), thickness=cv2.FILLED)

# 5. Invert Back
# Turn "Solid White Blobs" back into "Solid Black Blocks" on White.
solid_map = cv2.bitwise_not(inverted_img)

# 6. Save
output_path = 'solid_cave.png'
cv2.imwrite(output_path, solid_map)
print(f"Success! Saved solid map to {output_path}")
