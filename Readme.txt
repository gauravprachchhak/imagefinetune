Attempted first part of 5 question that is to generate HTML of images along with their classification scores

There are two files having code and their functionalities are mentioned below:

1. imagenet_finetune.py

This file was provided by the instructer and it is used for training the model. I have added some lines of code to train this model on GPU because it would take more time to run on CPU.

I Ran this on Google Cloud Platform and you can see some stats below:
OS Image: Debian customized for Deep Learning for Pytorch
GPU: NVIDIA Tesla P100 Workstation grade
Training Time:
87 Minutes

2. loadprint.py

This file contains code created by me to generate HTML from testing set which is of 10,000 images.
I have modified this code to run on CPU because as we did not have run this code for all 10,000 images and the output file would be very big to upload on coursys.

Also I have tested on model that was trained as given by instructor that is for 10 epochs and batch size 4 and I am getting 61% as accuracy on this model. And the size of model is around 94mb so I can not upload it.


Steps to run:

1. Train the model on GPU and then save it
	
   If you want to train on CPU then you will have to remove line 1,2,3 mentioned in the imagenet_finetune.py

   Command: python3 imagenet_finetune.py

2. Test the model on CPU from the saved model

   If you want to run GPU for all 10,000 images remove the breaking condition from the code. Mention cuda() where tensor processing is done and mention img.cpu().numpy() for normalizing image because numpy can not figure out cuda tensor

   Command: python3 loadprint.py

Screenshot of the HTML table is included in the report.pdf as well

