import cv2
import matplotlib.pyplot as plt

# Convert to grayscale
img = cv2.imread(r"C:\Users\Suriya Prakash\Desktop\Wallpapers\All mighty shogun.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur filter
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Left edge detection (Sobel X)
left_edge = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Top edge detection (Sobel Y)
top_edge = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Display results
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(blur, cmap='gray')
plt.title("Blur Filter")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(left_edge, cmap='gray')
plt.title("Left Edge Filter")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(top_edge, cmap='gray')
plt.title("Top Edge Filter")
plt.axis("off")

plt.show()
