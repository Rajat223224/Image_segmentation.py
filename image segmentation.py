import cv2

# Load the image
img = cv2.imread("C:\\Users\\rajat\\Desktop\\2.jpg")

# Convert the image to a 3-channel image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define the number of clusters and the number of iterations
num_clusters = 100
num_iterations = 10

# Create an instance of the SLIC algorithm
slic = cv2.ximgproc.createSuperpixelSLIC(img, cv2.ximgproc.SLIC, 10, 0.1)

# Perform the SLIC algorithm
slic.iterate(num_iterations)

# Obtain the labels for each pixel
labels = slic.getLabels()

# Create an output image with the segmented regions
output_img = slic.getLabelContourMask(False)
output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)

# Overlay the labels on the original image
output_img = output_img * 0.5 + img * 0.5

# Display the output image
cv2.imshow("Segmented Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
