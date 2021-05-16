1) Preprocessing the images: 
	- What Datatype? 
	- Are the intensity distributions comparable in every image?  
	- Find a transform to properly displays the images

2) Processing the images:
	- Find a proper TH-value for binarizing the images (Otsu if distribution hast two seperated peaks only)
	- Do the segmentation in connected components
	- implement certain criteria for filtering the structures (case activated and case resting) 
	- implement evaluation of parameters (case activated and case resting) 

3) User Interface: 
	- Visual Feedback of counted structures
	- User interaction: Remove marked structures
	- ROI Selecter: Cut tool: Use Input image Select ROI Perform processing
	- Session View: Load multiple images and store the session in a csv file
	- Session Statistic (optional)  
				