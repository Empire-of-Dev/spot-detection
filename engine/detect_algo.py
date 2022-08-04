from this import s
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO
class Processor():
	
	# initialize parameters with default values
	def __init__(self, dim):
		
		self.image = None # current image
		self.auto_mode = True # spot detection mode (auto vs manual)
		self.dim = dim # dimensions of displayed image
		
		# checkbox states for GUI controls for display preferences
		self.grid_checked = True
		self.intensity_checked = True
		self.leakage_checked = True

		# default parameters for image processing (most can be set from the GUI)
		self.threshold_offset = 40
		self.Nx_val = 7
		self.Ny_val = 4
		self.radius_val = 30
		self.leakage_val = 5
		self.min_area = 5000
		
		self.expected_x_map = None # store calculated x-coordinates of all spots
		self.expected_y_map = None # store calculated y-coordinates of all spots
		self.expected_r_map = None # store calculated radii of all spots
		self.Fs_arr = None  # store calculated intensities of all spots
	
	# initialize parameters with specified values
	def __init__(self, dim, threshold_offset, Nx_val, Ny_val, radius_val, 
		leakage_val, grid_checked, intensity_checked, leakage_checked):
		
		self.image = None
		self.auto_mode = True
		self.dim = dim
		
		self.grid_checked = grid_checked
		self.intensity_checked = intensity_checked
		self.leakage_checked = leakage_checked
		
		self.threshold_offset = threshold_offset
		self.Nx_val = Nx_val
		self.Ny_val = Ny_val
		self.radius_val = radius_val
		self.leakage_val = leakage_val
		self.min_area = 5000
		
		self.expected_x_map = None
		self.expected_y_map = None
		self.expected_r_map = None
		self.Fs_arr = None
	
	# return locations and sizes of detected circles
	def detect_circles(self, image):

		cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		circle_info = []

		# find centers 
		for cnt in cnts:
			(cX, cY), r = cv2.minEnclosingCircle(cnt)
			area = cv2.contourArea(cnt)
			circle_info.append([int(cX), int(cY), int(r), area])
			
		if len(circle_info) == 0:
			return None

		# filter out too small or too large circles using z-scores, as
		# those circles will have large magnitude of z-scores
		max_z = 0.3
		info = np.array(circle_info)
		z_r = (info[:, 2] - self.radius_val)/self.radius_val

		# remove invalid circles from the list
		circle_info_tmp = []
		for i in range(0, len(circle_info)):
			info = circle_info[i]
			cX = info[0]
			cY = info[1]
			r = info[2]
			area = info[3]
			if np.abs(z_r[i]) < max_z:
				circle_info_tmp.append([cX, cY, r])
				
		circle_info = circle_info_tmp
		
		return circle_info
	
	# find bright markers at 4 corners
	def localizeMarkers(self, circle_info):
		min_L1_dist = 100000
		max_L1_dist = 0
		
		top_left = (0, 0)
		top_right = (0, 0)
		bottom_left = (0, 0)
		bottom_right = (0, 0)
		
		for n in range(0, len(circle_info)): # loop each circle
			info = circle_info[n]
			cX = info[0]
			cY = info[1]
			r = info[2]
			
			if min_L1_dist > cX + cY:
				min_L1_dist = cX + cY
				top_left = (cX, cY)
			
			if max_L1_dist < cX + cY:
				max_L1_dist = cX + cY
				bottom_right = (cX, cY)
				
		min_L1_dist = 100000
		max_L1_dist = 0
				
		for n in range(0, len(circle_info)): # loop each circle
			info = circle_info[n]
			cX = info[0]
			cY = info[1]
			r = info[2]
			
			tmp_dist = np.abs(cX - bottom_right[0]) + np.abs(cY - top_left[1])
			if min_L1_dist > tmp_dist:
				min_L1_dist = tmp_dist
				top_right = (cX, cY)
			
			if max_L1_dist < tmp_dist:
				max_L1_dist = tmp_dist
				bottom_left = (cX, cY)
				
		return top_left, top_right, bottom_left, bottom_right
	
	# TODO
	def localizeSpots_manual(self):
		mask = self.image.copy()
		return mask
	
	# determine locations of both dark and bright spots and calculate their intensities
	def localizeSpots_auto(self):
		mask = self.image.copy()
		
		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (3, 3), 0)

		thresh = cv2.threshold(blurred, self.thresh_val + self.threshold_offset, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.erode(thresh, None, iterations = 2)
		thresh = cv2.dilate(thresh, None, iterations = 2)

		# do not perform any further processing if no potential spots are detected
		circle_info = self.detect_circles(thresh)
		if circle_info is None:
			mask = cv2.resize(mask, (self.dim[0], self.dim[1]))
			return mask

		top_left, top_right, bottom_left, bottom_right = self.localizeMarkers(circle_info)
		
		v1 = [bottom_left[0] - top_left[0], bottom_left[1] - top_left[1]]
		v2 = [top_right[0] - top_left[0], top_right[1] - top_left[1]]
		v3 = [bottom_right[0] - top_right[0], bottom_right[1] - top_right[1]]
		v4 = [bottom_right[0] - bottom_left[0], bottom_right[1] - bottom_left[1]]
	
		magnitude_1 = np.linalg.norm(v1)*np.linalg.norm(v2)
		magnitude_2 = np.linalg.norm(v3)*np.linalg.norm(v4)
		
		if magnitude_1 < self.radius_val or magnitude_2 < self.radius_val:
			mask = cv2.resize(mask, (self.dim[0], self.dim[1]))
			return mask
		
		angle_1 = np.arccos(np.dot(v1, v2) / magnitude_1)
		angle_2 = np.arccos(np.dot(v3, v4) / magnitude_2)
		some_markers_missing = np.abs(angle_1 - np.pi/2) > 0.15 or np.abs(angle_2 - np.pi/2) > 0.15
		
		if some_markers_missing:
			mask = cv2.resize(mask, (self.dim[0], self.dim[1]))
			return mask
		
		w = ((top_right[0] - top_left[0]) + (bottom_right[0] - bottom_left[0]))/2
		h = ((bottom_left[1] - top_left[1]) + (bottom_right[1] - top_right[1]))/2
		delta_x = 1.0*w / (self.Nx_val-1)
		delta_y = 1.0*h / (self.Ny_val-1)
		x = top_left[0] - delta_x/2
		y = top_left[1] - delta_y/2
		
		max_r = np.sqrt((delta_x/2)**2 + (delta_y/2)**2)
		
		if self.grid_checked:
			
			for i in range(self.Nx_val + 1):
				cv2.line(mask, (int(x + i*delta_x), int(y)), (int(x + i*delta_x), int(y + h + delta_y)), (255, 255, 0), 3)
				
			for j in range(self.Ny_val + 1):
				cv2.line(mask, (int(x), int(y + j*delta_y)), (int(x + w + delta_x), int(y + j*delta_y)), (255, 255, 0), 3)

		max_r = np.sqrt((delta_x/2)**2 + (delta_y/2)**2)
					
		self.expected_x_map = np.zeros((self.Ny_val, self.Nx_val))
		self.expected_y_map = np.zeros((self.Ny_val, self.Nx_val))
		self.expected_r_map = self.radius_val * np.ones((self.Ny_val, self.Nx_val))
		self.bright_spot_exist = np.zeros((self.Ny_val, self.Nx_val))

		for i in range(self.Ny_val):
			for j in range(self.Nx_val):
				tmp_point_x = x + (1.0*j + 1/2)*delta_x
				tmp_point_y = y + (1.0*i + 1/2)*delta_y

				self.expected_x_map[i, j] = tmp_point_x
				self.expected_y_map[i, j] = tmp_point_y

		# mark bright spots as green and dark spots as red
		for n in range(0, len(circle_info)): # loop each circle
			info = circle_info[n]
			cX = info[0]
			cY = info[1]
			r = info[2]

			for i in range(self.Ny_val):
				for j in range(self.Nx_val):
					if (self.expected_x_map[i, j] - cX)**2 + (self.expected_y_map[i, j] - cY)**2 < max_r**2:
						self.bright_spot_exist[i, j] = 1
						self.expected_x_map[i, j] = cX
						self.expected_y_map[i, j] = cY
						self.expected_r_map[i, j] = r
		
		self.expected_x_map = self.expected_x_map.astype(int)
		self.expected_y_map = self.expected_y_map.astype(int)
		self.expected_r_map = self.expected_r_map.astype(int)
	
		img_B = self.image[:, :, 0]
		img_G = self.image[:, :, 1]
		img_R = self.image[:, :, 2]
		img_gray = 1.0 * (img_B + img_G)/2
		
		self.Fs_arr = np.zeros((self.Ny_val, self.Nx_val))
		
		start_x = int(delta_x/2)
		end_x = int(delta_x/2 + delta_x)
		start_y = int(delta_y/2)
		end_y = int(delta_y/2 + delta_y)

		for i in range(self.Ny_val):
			for j in range(self.Nx_val):
				
				crop_start_x = np.maximum(int(self.expected_x_map[i, j] - delta_x), 0)
				crop_end_x = np.minimum(int(self.expected_x_map[i, j] + delta_x), self.image.shape[1])
				crop_start_y = np.maximum(int(self.expected_y_map[i, j] - delta_y), 0)
				crop_end_y = np.minimum(int(self.expected_y_map[i, j] + delta_y), self.image.shape[0])
				
				patch = img_gray[crop_start_y:crop_end_y, crop_start_x:crop_end_x].copy()

				y_grid, x_grid = np.ogrid[0:patch.shape[0], 0:patch.shape[1]]
				
				Ts_mask = (x_grid - delta_x)**2 + (y_grid - delta_y)**2 <= self.expected_r_map[i, j]**2
				T1_mask = (x_grid - delta_x)**2 + (y_grid - delta_y)**2 <= (self.expected_r_map[i, j] + self.leakage_val)**2
				T2_mask = np.zeros_like(patch)
				T2_mask[start_y:end_y+1, start_x:end_x+1] = 1
				T2_mask = T2_mask - T1_mask.astype(int)
				T2_mask[T2_mask < 0] = 0
				
				As = np.sum(Ts_mask)
				A1 = np.sum(T1_mask)
				A2 = int(delta_x * delta_y) - A1
				
				T1 = np.sum(patch[T1_mask > 0])
				T02 = np.sum(patch[T2_mask > 0])
				A2 = np.maximum(0.00001, A2)
				T01 = (T02 * A1)/A2
				As = np.maximum(0.00001, As)
				Fs = np.maximum((T1 - T01)/As, 0)
				self.Fs_arr[i, j] = Fs
				
				x_scaled = self.expected_x_map[i, j]
				y_scaled = self.expected_y_map[i, j]
				r_scaled = self.expected_r_map[i, j]
			
				if self.bright_spot_exist[i, j] == 1:
					cv2.circle(mask, (x_scaled, y_scaled), r_scaled, (0, 255, 0), 5)
					cv2.drawMarker(mask, (x_scaled, y_scaled), (0, 255, 0), cv2.MARKER_DIAMOND, 20, 5)
					if self.intensity_checked:
						cv2.putText(mask, '{:.2f}'.format(self.Fs_arr[i, j]), 
						(x_scaled - 50, y_scaled - r_scaled - 5), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
				else:
					cv2.circle(mask, (x_scaled, y_scaled), r_scaled, (0, 0, 255), 5)
					cv2.drawMarker(mask, (x_scaled, y_scaled), (0, 0, 255), cv2.MARKER_DIAMOND, 20, 5)
					if self.intensity_checked:
						cv2.putText(mask, '{:.2f}'.format(self.Fs_arr[i, j]), 
						(x_scaled - 50, y_scaled - r_scaled - 5), 
						cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
				
				if self.leakage_checked:		
					cv2.circle(mask, (x_scaled, y_scaled), 
					r_scaled + int(self.leakage_val/2), (255, 150, 255), self.leakage_val)
		
		# Draw preview of spot size
		# cv2.circle(mask, (mask.shape[1] - 75, mask.shape[0] - 75), self.radius_val, (255, 255, 0), 3)
		# cv2.putText(mask, "Spot size:", (mask.shape[1] - 275, mask.shape[0] - 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
		
		mask = cv2.resize(mask, (self.dim[0], self.dim[1]))
					
		return mask
	
	# drw crop region interactively as rectangle
	def paintCropRegion(self, image, begin, end):
		mask = image.copy()
		mask = cv2.resize(mask, (self.dim[0], self.dim[1]))
		
		if np.abs(end.x() - begin.x()) > 0 and np.abs(end.y() - begin.y()) > 0:
			begin_x = begin.x()
			begin_y = begin.y()
			end_x = end.x()
			end_y = end.y()
			cv2.rectangle(mask, (begin_x, begin_y), (end_x, end_y), (255, 255, 0), 2)
			
		return mask

	# load a new image from a given file path
	def loadNewImage(self, img):
		image_orig = img
		gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (3, 3), 0)
		self.thresh_val, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		self.setImage(image_orig)
		
		return image_orig
	
	# set the current image
	def setImage(self, image):
		self.image = image
	
	# retrieve updated image processing parameter from the main GUI
	def updateParams(self, threshold_offset, Nx_val, Ny_val, 
			radius_val, leakage_val, grid_checked, intensity_checked, leakage_checked):
		self.threshold_offset = threshold_offset
		self.Nx_val = Nx_val
		self.Ny_val = Ny_val
		self.radius_val = radius_val
		self.leakage_val = leakage_val
		
		self.grid_checked = grid_checked
		self.intensity_checked = intensity_checked
		self.leakage_checked = leakage_checked
	
	# return processed results, including spot locations, sizes, types, and intensities
	def getProcessedResults(self):
		return self.expected_x_map, self.expected_y_map, self.expected_r_map, self.bright_spot_exist, self.Fs_arr

def import_img(img):

	# Default parameters for image processing (most can be set from the GUI)
	DEFAULT_DIM = (900, 600)
	DEFAULT_Nx_val = 7
	DEFAULT_Ny_val = 4
	DEFAULT_RADIUS_VAL = 75
	DEFAULT_LEAKAGE_VAL = 5
	DEFAULT_MIN_AREA = 5000
	DEFAULT_THRESH_OFFSET = 0
	
	# Set parameter values here
    
    
	dim = DEFAULT_DIM
	threshold_offset = DEFAULT_THRESH_OFFSET
	Nx_val = DEFAULT_Nx_val
	Ny_val = DEFAULT_Ny_val
	radius_val = DEFAULT_RADIUS_VAL
	leakage_val = DEFAULT_LEAKAGE_VAL
	grid_checked = True
	intensity_checked = True
	leakage_checked = True
	
	# create processor instance for handling image processing tasks
	pc = Processor(dim, threshold_offset, Nx_val, Ny_val, radius_val, 
		leakage_val, grid_checked, intensity_checked, leakage_checked)
	
	# load an image from a given file path
	# and return an image prior to processing
	image_orig = pc.loadNewImage(img)
	
	# localize spots automatically and annotate the image
	mask = pc.localizeSpots_auto()
	b = (base64.b64encode(mask)).decode()

	mask_img = Image.fromarray(mask)	


	left = 0
	top = 120
	right = 510
	bottom = 400
	mask_img = mask_img.crop((left, top, right, bottom)) 
	
	buffered = BytesIO()
	mask_img.save(buffered, format="JPEG")
	img_str = (base64.b64encode(buffered.getvalue())).decode()  
	
	mask_img.save("test.png")


	
	return img_str
    	
	
	
	

	# display the annotated image
	#cv2.imshow('image', mask)
	#cv2.waitKey(0)
