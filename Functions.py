import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import os
from tkinter import font


# Constants for card size
CARD_MAX_AREA = 1000000   #Max area size might get really high with the camera view depending on the distance
CARD_MIN_AREA = 10000
THRESH_LVL = 140  # scale to the luminosity of your test environnement


class Template:
    """Structure to store information about template images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.rank = "Placeholder"
        self.value = 0

def Load_Templates(filepath):
    """
    Loads card templates (rank and suit images) from a specified directory.
    Each template is assigned a rank, suit, and value based on standard card rules.

    Args:
        filepath (str): Path to the directory containing card template images.

    Returns:
        list: A list of Template objects, each representing a card template with
              attributes such as rank, suit, value, and the associated image.
    """
    # Define ranks and suits
    ranks = ["A", "2", "3", "4", "5",
             "6", "7", "8", "9", "10",
             "J", "Q", "K"]
    suits = ["H", "S", "D", "C"]

    # List to store all template objects
    templates = []

    # Iterate over each suit
    for suit in suits:
        # Iterate over each rank
        for rank in ranks:
            # Initialize a Template object
            t_object = Template()
            t_object.rank = f"{rank}{suit}"  # Set the rank and suit as a combined identifier

            # Assign card value based on the rank
            if rank in ['J', 'Q', 'K']:  # Face cards have a value of 10
                t_object.value = 10
            elif rank == 'A':  # Aces have a value of 11
                t_object.value = 11
            else:  # Numeric cards have their rank as the value
                t_object.value = int(rank)

            # Construct the filename for the card template image
            filename = f"{rank}{suit}.jpg"
            img_path = os.path.join(filepath, filename)  # Full path to the image

            # Load the template image and handle potential errors
            t_object.img = cv2.imread(img_path)
            if t_object.img is None:
                print(f"Warning: Image {img_path} could not be loaded.")  # Warn if the image is missing

            # Add the Template object to the list
            templates.append(t_object)

    # Return the list of all loaded templates
    return templates




class Card:
    """Structure to store information about detected cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.img = [] # 200x300, flattened, grayed, blurred image
        self.rank = "Unknown"  # Detected rank of the card
        self.value = 0 # Detected value of the card


def Process_image(image, isCard=0):
    """
    Preprocesses the input image for further processing by converting it to grayscale,
    applying Gaussian blur, and thresholding.

    Args:
        image (ndarray): The input image, either in grayscale or color.
        isCard (int, optional): Flag indicating if the image contains a card (1 for card, 0 otherwise).
                                Defaults to 0.

    Returns:
        ndarray: A binary thresholded image.
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Image is already in grayscale

    # Apply Gaussian blur to reduce noise and smoothen the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Determine the image dimensions and select a central point for analysis
    height, width = gray.shape
    x = width // 2  # Horizontal center
    y = height - 7  # Near the bottom of the image

    # Set threshold level dynamically if processing a card image
    if isCard == 1:
        white_pixel_intensity = gray[y, x]  # Intensity of a pixel near the bottom center
        dynamic_thresh_lvl = max(white_pixel_intensity - 15, 0)  # Adjust threshold to handle card brightness
        thresh_lvl = dynamic_thresh_lvl
    else:
        thresh_lvl = THRESH_LVL  # Use a predefined threshold level for non-card images

    # Apply binary thresholding to create a binary image
    retval, thresh = cv2.threshold(blur, thresh_lvl, 255, cv2.THRESH_BINARY)

    return thresh

def Find_cards(image):
    """
    Identifies potential card contours in an image and draws valid contours.

    Args:
        image (ndarray): The input image, either grayscale or color.

    Returns:
        tuple:
            - valid_contours (list): A list of contours that match the criteria for being cards.
            - output_image (ndarray): A copy of the input image with valid contours drawn.
    """
    # Preprocess the input image to obtain a binary thresholded image
    img = Process_image(image)

    # Find contours and their hierarchy from the processed binary image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []  # Initialize a list to store contours that represent cards
    output_image = image.copy()  # Create a copy of the input image for visualization

    # Return early if no contours are found
    if len(contours) == 0:
        return valid_contours, output_image

    # Iterate through all detected contours
    for i, contour in enumerate(contours):
        size = cv2.contourArea(contour)  # Calculate the area of the contour
        
        # Check if the contour's area is within the valid range for cards
        if not (CARD_MIN_AREA < size < CARD_MAX_AREA):
            continue
        
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)  # Calculate the perimeter of the contour
        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)  # Approximate the contour

        # Check if the approximated contour is a quadrilateral and not nested
        if len(approx) == 4 and hierarchy[0][i][3] == -1:
            valid_contours.append(contour)  # Add the contour to the valid list
            # Draw the valid contour on the output image in green
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


def compute_diff_score(image1, image2):
    """
    Computes a difference score between two images by processing them and summing their absolute differences.

    Args:
        image1 (ndarray): The first input image.
        image2 (ndarray): The second input image.

    Returns:
        int: A score representing the sum of absolute differences between the two images.
             A lower score indicates higher similarity.
    """
    # Preprocess both images to generate binary thresholded versions
    img1 = Process_image(image1, 1)  # Process the first image with card-specific settings
    img2 = Process_image(image2, 1)  # Process the second image with card-specific settings

    # Compute the absolute difference between the two processed images
    diff = cv2.absdiff(img1, img2)

    # Sum the pixel differences to calculate a similarity score
    # A lower sum indicates the images are more similar
    return np.sum(diff)




def Match_Cards(cards, image, templates):
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
        value_best_match = 0
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
                value_best_match = template.value

       

        # After checking all family ranks and suits, assign the best matches
        if best_match is not None:
            processed_card.rank = best_match
            processed_card.value = value_best_match
        

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


def Save_template_cards(card, dest_folder, unique_name):
    """
    Saves a card image to the specified folder with a unique filename.

    Args:
        card (ndarray): The card image to be saved.
        dest_folder (str): The destination folder where the image will be stored.
        unique_name (str): A unique identifier for naming the saved file.

    Returns:
        None
    """
    # Construct the full path for the output file using the unique name and destination folder
    filename = os.path.join(dest_folder, f'{unique_name}.jpg')

    # Save the card image as a .jpg file
    cv2.imwrite(filename, card)




def divide_into_zones(frame, scorep1=0, scorep2=0, scorep3=0, scorep4=0):
    """
    Divides the input frame into four zones: top-left, top-right, bottom-left, and bottom-right.
    Annotates each zone with the corresponding score and identifies the zone with the highest score
    or marks it as a tie. Adds "Lost" or "blackjack!" text for specific conditions.

    Args:
        frame (ndarray): The input video frame.
        scorep1, scorep2, scorep3, scorep4 (int, optional): Scores for each of the four zones. Defaults to 0.

    Returns:
        tuple:
            - frame (ndarray): The modified frame with annotated zones and status text.
            - zones (dict): A dictionary with keys for each zone and their respective cropped areas.
    """

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Calculate midpoints to define the boundaries of the zones
    mid_x, mid_y = width // 2, height // 2

    # Define zones as regions of the frame
    zones = {
        "top_left": frame[0:mid_y, 0:mid_x],
        "top_right": frame[0:mid_y, mid_x:width],
        "bottom_left": frame[mid_y:height, 0:mid_x],
        "bottom_right": frame[mid_y:height, mid_x:width]
    }

    # Draw dividing lines to visually separate zones
    cv2.line(frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)  # Vertical line
    cv2.line(frame, (0, mid_y), (width, mid_y), (0, 255, 0), 2)   # Horizontal line

    # Annotate each zone with its respective player's score
    titles = {
        "top_left": f"Player 1 : {scorep1}",
        "top_right": f"Player 2 : {scorep2}",
        "bottom_left": f"Player 3 : {scorep3}",
        "bottom_right": f"Player 4 : {scorep4}"
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 255, 0)  # Green
    thickness = 2

    # Draw the titles on the frame
    cv2.putText(frame, titles["top_left"], (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, titles["top_right"], (mid_x + 10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, titles["bottom_left"], (10, mid_y + 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, titles["bottom_right"], (mid_x + 10, mid_y + 30), font, font_scale, color, thickness, cv2.LINE_AA)

    # Compute the highest score and determine the winning zone
    scores = [scorep1, scorep2, scorep3, scorep4]
    zones_names = ["top_left", "top_right", "bottom_left", "bottom_right"]

    # Adjust scores, treating those over 21 as invalid (-inf)
    valid_scores = [score if score <= 21 else float('-inf') for score in scores]

    # Check for a draw or a winning zone
    if max(valid_scores) == float('-inf'):  # All scores exceed 21
        center = (mid_x, mid_y)
        status = "Draw"
    else:
        max_score_index = valid_scores.index(max(valid_scores))
        winning_zone = zones_names[max_score_index]
        status = "Winning"

    # Handle ties for the highest score
    highest_scores = sorted(valid_scores, reverse=True)
    if max(valid_scores) == 0:  # No valid scores
        center = (mid_x, mid_y)
        status = ""
    elif highest_scores[0] == highest_scores[1]:  # Tie detected
        center = (mid_x, mid_y)
        status = "Draw"
    elif winning_zone == "top_left":
        center = (mid_x // 2, mid_y // 2)
    elif winning_zone == "top_right":
        center = (mid_x + mid_x // 2, mid_y // 2)
    elif winning_zone == "bottom_left":
        center = (mid_x // 2, mid_y + mid_y // 2)
    elif winning_zone == "bottom_right":
        center = (mid_x + mid_x // 2, mid_y + mid_y // 2)

    # Annotate zones with "Bust" if scores exceed 21 or "blackjack!" for scores of 21
    for i, score in enumerate(scores):
        if score > 21:
            zone_center = (mid_x // 2, mid_y // 2) if zones_names[i] == "top_left" else (
                (mid_x + mid_x // 2, mid_y // 2) if zones_names[i] == "top_right" else (
                    (mid_x // 2, mid_y + mid_y // 2) if zones_names[i] == "bottom_left" else (mid_x + mid_x // 2, mid_y + mid_y // 2)))
            cv2.putText(frame, "Bust", zone_center, font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        elif score == 21:
            zone_center = (mid_x // 2, (mid_y // 2) + 50) if zones_names[i] == "top_left" else (
                (mid_x + mid_x // 2, mid_y // 2 + 50) if zones_names[i] == "top_right" else (
                    (mid_x // 2, mid_y + mid_y // 2 + 50) if zones_names[i] == "bottom_left" else (mid_x + mid_x // 2, mid_y + mid_y // 2 + 50)))
            cv2.putText(frame, "blackjack!", zone_center, font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # Add "winning" or "Draw" status to the center of the frame or the winning zone
    if status == "Winning":
        cv2.putText(frame, status, center, font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    elif status == "Draw":
        cv2.putText(frame, status, center, font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    return frame, zones



def determine_zone(frame, card):
    """
    Determines in which zone the center of the card is located.
    
    Args:
        frame: The input video frame (used to calculate midpoints).
        card: A card object that contains the center attribute.
    
    Returns:
        The zone name as a string ("top_left", "top_right", "bottom_left", "bottom_right").
    """
    player_1 = []
    player_2 = []
    player_3 = []
    player_4 = []
    height, width, _ = frame.shape  # Get frame dimensions

    # Calculate midpoints of the frame
    mid_x, mid_y = width // 2, height // 2

    # Get the center of the card (card.center should be a tuple or list like (x, y))
    center = card.center  # Assuming card.center is a (x, y) tuple

    # Determine which zone the card center belongs to
    if center[0] < mid_x and center[1] < mid_y:
        player_1.append(card)
    elif center[0] >= mid_x and center[1] < mid_y:
        player_2.append(card)
    elif center[0] < mid_x and center[1] >= mid_y:
        player_3.append(card)
    else:
        player_4.append(card)

    return player_1, player_2, player_3, player_4    

def calculate_score(cards):
    score = 0   
    for card in cards:
        value = card.value
        score += int(value)

    return score

