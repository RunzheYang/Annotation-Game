import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style='ticks', palette='Set2')
# Visualization Tools
def vis_data(data):
	_, ax = plt.subplots(2, len(data), figsize=(len(data), 2));
	for i in xrange(len(data)):
		name = ""
		ansname = ""
		(x, y) = data[i]
		for j in xrange(3):
			name += str(x[j])
		ansname = str(y)
		img = mpimg.imread("imgs/"+ name +".png")
		ans = mpimg.imread("imgs/"+ ansname +".png")
		if len(data) > 1:
			ax[0, i].imshow(img); ax[0, i].axis('off');
			ax[1, i].imshow(ans); ax[1, i].axis('off');	
		else:
			ax[0].imshow(img); ax[0].axis('off');
			ax[1].imshow(ans); ax[1].axis('off');	

# Initialize the Student Belief: 
def l1normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0: return v
    else: return v / norm