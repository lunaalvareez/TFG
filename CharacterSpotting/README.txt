

Training:

Create main folder "Characterspotting" on your computer.

Create the subfolders "/wordsCompressed", "/wordsCompressedSubset40K", "/txts", "/trainEvalSet", "/testSet"

Get "words", "xml", "trainset.txt" from IAMHandwriting dataset. Place them in your root directory (/Characterspotting)

Before each run. Make sure to read the comments and change the paths as necessary. Run them in accordance with the order.
Run:
1. Compressor.py: and make sure /wordsCompressed is filled with around 1.6GB. If you can open the folder and look through that's good but it's not needed if it's not possible.

2. AnnotationGenerator.py: Now the /wordsCompressed should have annotation files accompanying each image. wordsCompressed should have an approximate size of 1.8GB.

3. Optional: RemoveSmallImages.py: Removes all images and annotation files with width or height smaller than 10 in /wordsCompressed. Also removes images with errors being opened. Decreases /wordscompressed by 5%.

4. Optional: RemoveSingleDigits.py: Removes all Images and annotation files with single digits in /wordscompressed. Decreases /wordscompressed by 11%.

5. Subset40KGenerator.py: Now the /wordsCompressedSubset40k should have 40k random images from /wordsCompressed with accompanying annotation files. This folder should be able to be opened and inspected. Approximate size 444 MB

6. Optional: VerifySubsetSize.py: Verifies that the size is correct in /wordsCompressedSubset40k

7. TrainEvalTestSplit.py: Should split the /wordsCompressedSubset40K into /trainEvalSet and /Testset. With approximate size /trainEvalset: 16.5 MB and /Testset: 428 MB.


8. Google Colab:
Create a new directory with "/Characterspotting" as name in your main drive directory. Make sure you're logged in to the same account as the one you will do the training with. In this create "/data" and put "/trainEvalSet" Also create a folder in /Characterspotting named "TrainingOutput".

Final drive directory map should be as follows:
Characterspotting
|
|--------/data
|	|
|	|--------|-/trainEvalset
|	|	 	|-All Images and Annotation files ex "m02-048-00-00.png"
|	|	
|	|--------|-/txts (Don't put this in yet. We need to fill it in first)
|			|Eval.txt (To be created)
|			|Train.txt (To be created)
|			|data_info_yaml (To be created)
|
|--------/TrainingOutput


9. Go to "https://colab.research.google.com/drive/1yyveuIVoKkmKEQWvI117dri69wsZ8Vbm?authuser=1#scrollTo=yWLGke23qCQy" and follow the instructions.

10. TxtFileGenerator.py: Should create Eval.txt, Train.txt and Test.txt in /txts !!Important to read the comments and change directories based on your drive path to "/Characterspotting".

11. ContaminationVerify.py: Make sure there are no common filenames between val, train and test.

12. Optional: Open "data_info.yaml" and change names, directories, nc(name count) etc as you see fit. Otherwise leave them as they are.

13. Copy "trainEvalSet" from /root into drive /data. You might have to compress the file to make it go faster. Use one of Google's inbuilt tools to unzip.
Copy "train.txt","val.txt" and "data_info.yaml" from /txts into drive /txts.

