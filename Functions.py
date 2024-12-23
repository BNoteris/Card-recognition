import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import os
from tkinter import font


# Constants for card size
CARD_MAX_AREA = 1000000   #Max area size might get really high with the camera view depending on the distance
CARD_MIN_AREA = 10000
# Width and height of card corner
CORNER_WIDTH = 45
CORNER_HEIGHT = 70
MATCH_VALUE = 500000
THRESH_LVL = 140


class Template:
    """Structure to store information about family rank images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.rank = "Placeholder"

def Load_Templates(filepath):    
    ranks = ["A", "2", "3", "4","5",
             "6", "7", "8", "9", "10",
             "J", "Q", "K"]
    suits = ["H", "S", "D", "C"]

    templates = []
    for suit in suits:
        for rank in ranks:
            t_object = Template()
            t_object.rank = f"{rank}{suit}"
            filename = f"{rank}{suit}.jpg"
            img_path = os.path.join(filepath, filename)
            
            # Load image and handle potential errors
            t_object.img = cv2.imread(img_path)
            if t_object.img is None:
                print(f"Warning: Image {img_path} could not be loaded.")

            templates.append(t_object)

    return templates        



class Card:
    """Structure to store information about detected cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.img = [] # 200x300, flattened, grayed, blurred image
        self.corner = [] # image of the corner of the card
        self.rank = "Unknown"  # Detected rank of the card
        self.suit = "Unknown" # Detected suit of the card


def Process_image(image):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    retval, thresh = cv2.threshold(blur,THRESH_LVL,255,cv2.THRESH_BINARY)
    return thresh

def Find_cards(image):

    img = Process_image(image)
     # Find contours and their hierarchy
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []  # List to store valid card contours
    output_image = image.copy()  # Copy of the input image for drawing contours

    # Return early if no contours are found
    if len(contours) == 0:
        return valid_contours, output_image
    
    for i, contour in enumerate(contours):
        size = cv2.contourArea(contour)  # Calculate contour area
        if not (CARD_MIN_AREA < size < CARD_MAX_AREA):  # Check if area is within valid range
            continue
        
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        
        # Only consider quadrilateral contours that are not nested
        if len(approx) == 4 and hierarchy[0][i][3] == -1 :
            valid_contours.append(contour)
            # Draw the valid contour on the output image
            cv2.drawContours(output_image, [contours[i]], -1, (0, 255, 0), 3)

    return valid_contours, output_image


def Process_Card(contour, image):
    """
    Extracts relevant features from a card's contour in the image.

    Args:
        contour (ndarray): The contour of the card.
        image (ndarray): The input image.

    Returns:
        Card: A Card object with processed attributes.
    """
    import numpy as np
    import cv2

    # Initialize the Card object
    card = Card()
    card.contour = contour  # Store the contour

    # Approximate the contour with a tighter threshold to handle rounded corners
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.015 * peri, True)

    #Adjust the card to the right angle 
    rect = cv2.minAreaRect(contour)    
    box = cv2.boxPoints(rect)
    approx = np.int64(box)

    # Sort the corner points to a consistent order (top-left, top-right, bottom-right, bottom-left)
    pts = approx.reshape(-1, 2)
    s = np.sum(pts, axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]       # Top-left
    rect[2] = pts[np.argmax(s)]       # Bottom-right
    rect[1] = pts[np.argmin(diff)]    # Top-right
    rect[3] = pts[np.argmax(diff)]    # Bottom-left

    card.corner_pts = rect

    # Get bounding rectangle dimensions
    x, y, card.width, card.height = cv2.boundingRect(contour)

    # Compute card center using the reordered corner points
    card.center = np.mean(rect, axis=0).astype(int).tolist()

    # Reshape and transform the card image
    card.img = Reshape_Card(image, rect, card.width, card.height)
    
    return card


def Reshape_Card(image, corner_pts, width, height):
    """
    Warps the card image into a standard 200x300 size.

    Args:
        image (ndarray): The input image.
        corner_pts (ndarray): Corner points of the card.
        width (int): Width of the bounding rectangle.
        height (int): Height of the bounding rectangle.

    Returns:
        ndarray: The warped and grayscaled card image.
    """
    
    # If corner_pts is a 2D array with 4 points and 2 coordinates each
    if corner_pts.shape == (4, 2):
        # Step 1: Sort points by y-coordinate (top-to-bottom)
        sorted_pts = corner_pts[np.argsort(corner_pts[:, 1])]

        # Step 2: Split into top and bottom pairs
        top_pts = sorted_pts[:2]
        bottom_pts = sorted_pts[2:]

        # Step 3: Sort top and bottom pairs by x-coordinate (left-to-right)
        top_left, top_right = top_pts[np.argsort(top_pts[:, 0])]
        bot_left, bot_right = bottom_pts[np.argsort(bottom_pts[:, 0])]

        # Step 4: Determine card orientation
        temp_rect = np.zeros((4, 2), dtype="float32")
        if width <= 0.8 * height:  # Vertical orientation
            temp_rect[:] = [top_left, top_right, bot_right, bot_left]
        elif width >= 1.2 * height:  # Horizontal orientation
            temp_rect[:] = [bot_left, top_left, top_right, bot_right]
        else:  # Diamond orientation
            temp_rect[:] = [top_left, top_right, bot_right, bot_left]  # Maintain consistent order


        # Define destination points for warping
        dst = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype="float32")

        # Perform perspective transformation and grayscale conversion
        M = cv2.getPerspectiveTransform(temp_rect, dst)
        warp = cv2.warpPerspective(image, M, (200, 300))
        
        return cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    else:
        print("Invalid corner_pts shape!")
        return None



def Get_Card_Corner(card_reshaped):
    """
    Extracts and zooms in on the corner of the reshaped card image.
    
    Args:
        card_reshaped (ndarray): Warped card image of size 200x300.
    
    Returns:
        ndarray: The zoomed-in corner of the card.
    """
    # Extract the top-left corner of the card image
    card_corner = card_reshaped[10:CORNER_HEIGHT, 10:CORNER_WIDTH]
    
    # Perform a 4x zoom on the corner
    card_corner_zoom = cv2.resize(card_corner, (0, 0), fx=4, fy=4)
    card_rank_zoom = card_corner_zoom[0:170, 0:160]
    card_suit_zoom = card_corner_zoom[130:280, 0:160]


    
    return card_corner_zoom,card_rank_zoom,card_suit_zoom

def process_zoom(zoom_region):
        # Preprocess the image
        blur = cv2.GaussianBlur(zoom_region, (3, 3), 0)
        edges = cv2.Canny(blur, 10, 50)
        kernel = np.ones((3, 3))
        dial = cv2.dilate(edges, kernel=kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dial, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by contour area

        # Check if contours exist and return the bounding box
        if sorted_contours:
            return cv2.boundingRect(sorted_contours[0])
        else:
            return None  # No bounding box found


def compute_diff_score(image1, image2):

        img1 = Process_image(image1)
        img2 = Process_image(image2)
           
        # Compute absolute difference
        diff = cv2.absdiff(img1, img2)

       

        # Sum the differences as a similarity score
        return np.sum(diff)




def Save_Corner(corner, image_name):
    """
    Saves the zoomed-in card corner to a file.
    
    Args:
        corner (ndarray): The corner image to save.
        image_name (str): Name for the saved image file.
    """
    # Construct the filename and save the corner image
    filename = f'Corners/{image_name}.jpg'
    cv2.imwrite(filename, corner)



def Match_Corners(cards, image, templates):
    """
    Matches detected cards with family ranks and suits based on corner similarity.

    Args:
        cards (list): List of contours representing detected cards.
        image (ndarray): The input image containing the cards.
        ranks (list): List of Family_ranks objects representing the family ranks.
        suits (list): List of Family_suits objects representing the family suits.

    Returns:
        list: A list of Card objects with identified ranks and suits.
    """
    matching_cards = []
    cards_found = []    

    # Loop through each detected card
    for card in cards:
        processed_card = Process_Card(card, image)  # Extract card features
        card_img = processed_card.img
        cards_found.append(processed_card)

        best_match = None  # To store the best match for the current card's rank
        
        best_diff_value = float('inf')  # Initialize with a large value for ranks
       

        # Compare the card corner with each family rank
        for template in templates:
            t_img = template.img
            if t_img is None:
                continue
            
            # Use compare_bounding_boxes_absdiff to get absdiff scores
            absdiff_scores = compute_diff_score(t_img, card_img)

            # Update the best match if a lower diff value is found
            if absdiff_scores < best_diff_value:
                best_diff_value = absdiff_scores
                best_match = template.rank

       

        # After checking all family ranks and suits, assign the best matches
        if best_match is not None:
            processed_card.rank = best_match
        

        matching_cards.append(processed_card)

    return matching_cards, cards_found




def draw_results(image, card):
    """
    Draws the detected card's rank and center point on the image.
    
    Args:
        image (ndarray): The input image to annotate.
        card (Card): The Card object containing detected attributes.
    
    Returns:
        ndarray: The annotated image.
    """
    # Extract card center coordinates
    x, y = card.center
    
    # Draw a blue circle at the card's center
    cv2.circle(image, (x, y), radius=25, color=(255, 0, 0), thickness=-1)

    # Write the detected rank near the card
    rank_name = card.rank

    cv2.putText(image, rank_name, (x - 60, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

    return image


def Save_template_cards(card, dest_folder,unique_name):
    filename = os.path.join(dest_folder, f'{unique_name}.jpg')
    cv2.imwrite(filename, card)

