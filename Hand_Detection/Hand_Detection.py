import cv2
import numpy as np
import matplotlib.pyplot as plt


def showplt(frame, gray, edges):
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Grayscale image
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    plt.title('Grayscale Image')
    plt.axis('off')

    # Edge-detected image
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    plt.title('Edge-detected Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def preprocessing_image(RGBImage):
    grayScale_image = cv2.cvtColor(RGBImage, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(grayScale_image, (5, 5), 0)

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # Thresholding using Otsu's method
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu's threshold value:", _)

    return grayScale_image, blurred, sharpened, binary


def KMean(image):
    # Convert color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image into a 2D array of pixels (rows) and 3 color values (RGB columns)
    pixel_vals = image.reshape((-1, 3))  # -1 means "unspecified dimension" (calculate automatically)

    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    k = 2

    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    # Map each pixel's label to its corresponding center color
    segmented_data = centers[labels.flatten()]

    # Reshape the segmented data back to the original image dimensions
    segmented_image = segmented_data.reshape(image.shape)

    plt.imshow(segmented_image)
    plt.show()

def detect_edges(grayScale_Image):
    # Sobel edge detection
    sobelX = cv2.Sobel(grayScale_Image, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(grayScale_Image, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    Combined = cv2.bitwise_or(sobelX, sobelY)
    # #####################
    # Remove weak edges using hysteresis thresholding
    strong_edges = cv2.threshold(Combined, 100, 255, cv2.THRESH_BINARY)[1]
    weak_edges = cv2.threshold(Combined, 50, 255, cv2.THRESH_BINARY)[1]

    final_edges = strong_edges.copy()
    idx = (weak_edges > 0) & (strong_edges == 0)
    final_edges[idx] = 0

    return final_edges


def match_edges(sobel_combined_captured, dataset_images, dataset_edges):
    best_match_index = -1
    best_match_score = float('-inf')
    for i, dataset_edge in enumerate(dataset_edges):
        result = cv2.matchTemplate(sobel_combined_captured, dataset_edge, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_match_score:
            best_match_score = max_val
            best_match_index = i
    if best_match_index == -1:
        print("No suitable match found.")
    else:
        print(f"Best match is Image {best_match_index + 1} with match score {best_match_score:.2f}")
        plt.imshow(cv2.cvtColor(dataset_images[best_match_index], cv2.COLOR_BGR2RGB))
        plt.title(f"Matched Image {best_match_index + 1}")
        plt.axis('off')
        plt.show()
def main():
    ip_webcam_url = "http://192.168.33.229:4747/video"  # mobile camera
    cap = cv2.VideoCapture(ip_webcam_url)

    # cap = cv2.VideoCapture(0)  # labtop camera

    # image = cv2.imread('3.jpg', 1)
    # frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    while True:
        ret, image = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Preprocess the image
        gray, blurred, sharpened, binary = preprocessing_image(image)
        # Detect edges in the captured image
        sobel_combined_captured = detect_edges(sharpened)
        # Combine images for display
        top_row = np.hstack((image, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
        bottom_row = np.hstack((cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(sobel_combined_captured, cv2.COLOR_GRAY2BGR)))
        combined = np.vstack((top_row, bottom_row))
        combined = cv2.resize(combined, (1024, 768))
        cv2.imshow('Hand Edge Detection', combined)
        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('s'):  # Press 's' to save the image
            dataset_images = [
                cv2.imread("C:/Users/M.HEPTON/Desktop/Newfolder(2)/DataSet/01.png"),
                cv2.imread("C:/Users/M.HEPTON/Desktop/Newfolder(2)/DataSet/02.png"),
                cv2.imread("C:/Users/M.HEPTON/Desktop/Newfolder(2)/DataSet/03.png"),
            ]
            dataset_edges = []
            for dataset_image in dataset_images:
                _, _, dataset_sharpened, _ = preprocessing_image(dataset_image)
                dataset_edges.append(detect_edges(dataset_sharpened))
            match_edges(sobel_combined_captured, dataset_images, dataset_edges)
            showplt(image, gray, sobel_combined_captured)
            print("image --->", image.shape)
            print("gray --->", gray.shape)
            print("binary --->", binary.shape)
            print("edges --->", sobel_combined_captured.shape)
            plt.figure(figsize=(10, 8))
            # Segmentation Step
            KMean(blurred)  # image = RGB   ,,, gray=GrayScale

    cap.release()
    cv2.destroyAllWindows()


# Run the main function
main()



