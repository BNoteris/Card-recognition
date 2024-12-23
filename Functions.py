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
        self.value = 0

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
            if rank in ['J', 'Q', 'K']:  
                    t_object.value = 10
            elif rank == 'A':  
                t_object.value = 11
            else:
                t_object.value = int(rank)

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
        self.value = 0 # Detected suit of the card


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


def compute_diff_score(image1, image2):

        img1 = Process_image(image1)
        img2 = Process_image(image2)
           
        # Compute absolute difference
        diff = cv2.absdiff(img1, img2)

       

        # Sum the differences as a similarity score
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


def Save_template_cards(card, dest_folder,unique_name):
    filename = os.path.join(dest_folder, f'{unique_name}.jpg')
    cv2.imwrite(filename, card)

import cv2

def divide_into_zones(frame, scorep1=0, scorep2=0, scorep3=0, scorep4=0):
    """
    Divides the frame into 4 zones: top-left, top-right, bottom-left, bottom-right.
    Also marks the zone with the highest score with the text "winning" in the center of that zone.
    
    Args:
        frame: The input video frame.
        scorep1, scorep2, scorep3, scorep4: Scores for each of the four zones.
    
    Returns:
        The modified frame with the zones and the "winning" text, and the zones dictionary.
    """
    height, width, _ = frame.shape  # Get frame dimensions

    # Calculate midpoints
    mid_x, mid_y = width // 2, height // 2

    # Define the zones
    zones = {
        "top_left": frame[0:mid_y, 0:mid_x],
        "top_right": frame[0:mid_y, mid_x:width],
        "bottom_left": frame[mid_y:height, 0:mid_x],
        "bottom_right": frame[mid_y:height, mid_x:width]
    }

    # Draw visual lines for the zones
    cv2.line(frame, (mid_x, 0), (mid_x, height), (0, 255, 0), 2)  # Vertical line
    cv2.line(frame, (0, mid_y), (width, mid_y), (0, 255, 0), 2)   # Horizontal line

    # Add titles for each zone
    titles = {
        "top_left": f"Player 1 : {scorep1} ",
        "top_right": f"Player 2 : {scorep2} ",
        "bottom_left": f"Player 3 : {scorep3} ",
        "bottom_right": f"Player 4 : {scorep4} "
    }

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 255, 0)  # Green
    thickness = 2

    # Add text for each zone
    cv2.putText(frame, titles["top_left"], (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, titles["top_right"], (mid_x + 10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, titles["bottom_left"], (10, mid_y + 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, titles["bottom_right"], (mid_x + 10, mid_y + 30), font, font_scale, color, thickness, cv2.LINE_AA)

    # Determine which zone has the highest score
    scores = [scorep1, scorep2, scorep3, scorep4]
    zones_names = ["top_left", "top_right", "bottom_left", "bottom_right"]

    # Filter scores greater than 21
    valid_scores = [score if score <= 21 else float('-inf') for score in scores]

    # Find the index of the highest valid score
    if max(valid_scores) == float('-inf'):
        # All scores are over 21
        center = (mid_x, mid_y)
        status = "Draw"
    else:
        max_score_index = valid_scores.index(max(valid_scores))
        winning_zone = zones_names[max_score_index]
        status = "Winning"

    # Handling ties
    highest_scores = sorted(valid_scores, reverse=True)

    # Determine the center coordinates of the winning zone
    if max(valid_scores) == 0:
        center = (mid_x, mid_y)
        status = ""
    elif highest_scores[0] == highest_scores[1]:
        center = (mid_x, mid_y)
        status = "Draw"
    elif winning_zone == "top_left":
        center = (mid_x // 2, mid_y // 2)
    elif winning_zone == "top_right":
        center = (mid_x + mid_x // 2, mid_y // 2)
    elif winning_zone == "bottom_left":
        center = (mid_x // 2, mid_y + mid_y // 2)
    elif winning_zone == "bottom_right":  # "bottom_right"
        center = (mid_x + mid_x // 2, mid_y + mid_y // 2)

    # Add "Lost" text to zones with scores over 21
    for i, score in enumerate(scores):
        if score > 21:
            zone_center = (mid_x // 2, mid_y // 2) if zones_names[i] == "top_left" else (
                (mid_x + mid_x // 2, mid_y // 2) if zones_names[i] == "top_right" else (
                    (mid_x // 2, mid_y + mid_y // 2) if zones_names[i] == "bottom_left" else (mid_x + mid_x // 2, mid_y + mid_y // 2)))
            cv2.putText(frame, "Lost", zone_center, font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # Add "winning" text to the winning zone
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

