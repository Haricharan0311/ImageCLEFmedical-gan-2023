# Dataset info
The benchmarking image data are the axial slices of 3D CT images of about 8000 lung tuberculosis patients. This particularly means that some of them may appear pretty “normal” whereas the others may contain certain lung lesions including the severe ones. These images are stored in the form of 8 bit/pixel PNG images with dimensions of 256x256 pixels.

The artificial slice images are 256x256 pixels in size. All of them were generated using the Diffuse Neural Networks.

The published development dataset for task includes 500 artificial images, 80 real images which were not used for training generative neural networks as well as 80 real images taken from the image set which has been used for training corresponding generative model.
Development dataset is available here: https://ctipub-my.sharepoint.com/:u:/g/personal/ana_dragulinescu_upb_ro/EUKCLoqAXqxJvnf4yGC_94YB4tbzieuZrap9QAsp3CO-gg?e=4zDL49

The test dataset was created in similar way. The only difference is that the two subsets of real images are mixed and no proportion of non-used and used ones has been disclosed. Thus, a total of 10,000 generated and 200 real images are provided. Test dataset is available here: https://ctipub-my.sharepoint.com/:u:/g/personal/ana_dragulinescu_upb_ro/EU6-Jx_yUfhGtJb4WDmSWhEBffiBA_mdFoDvDhMOCeBOaw?e=ffZOwL
