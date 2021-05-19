1) Preprocessing the images: 
	(done) What Datatype? 
	(done) Are the intensity distributions comparable in every image?  
	(done) Find a transform to properly displays the images

2) Processing the images:
	- Find a proper TH-value for binarizing the images (Otsu if distribution hast two seperated peaks only)
	(done) Do the segmentation in connected components
	(done) implement certain criteria for filtering the structures (case activated and case resting) 
	(done) implement evaluation of parameters (case activated and case resting) 

3) User Interface: 
	- Visual Feedback of counted structures
	- ROI Selecter: Cut tool: Use Input image Select ROI Perform processing
	- User interaction: Remove marked structures
	- User interaction: cut connected structures
	- Session View: Load multiple images and store the session in a csv file
	- Session Statistic  
				