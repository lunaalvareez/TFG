

Training:

Create main folder "Characterspotting" on your computer.

Create the subfolders "/sentencesCompressed", "/sentencesCompressedSubset40K", "/txts", "/trainEvalSet", "/testSet"

Get "words", "xml", "trainset.txt" from IAMHandwriting dataset. Place them in your root directory (/Characterspotting)

Before each run. Make sure to read the comments and change the paths as necessary. Run them in accordance with the order.
Run:
1. Compressor.py: and make sure /sentencesCompressed is filled with around 1.6GB. If you can open the folder and look through that's good but it's not needed if it's not possible.

2. AnnotationGenerator.py: Now the /sentencesCompressed should have annotation files accompanying each image. sentencesCompressed should have an approximate size of 1.8GB.

3. Optional: RemoveSmallImages.py: Removes all images and annotation files with width or height smaller than 10 in /sentencesCompressed. Also removes images with errors being opened. Decreases /wordscompressed by 5%.

4. Optional: RemoveSingleDigits.py: Removes all Images and annotation files with single digits in /sentencesCompressed. Decreases /sentencesCompressed by 11%.

5. Subset40KGenerator.py: Now the /sentencesCompressedSubset40k should have 40k random images from /sentencesCompressed with accompanying annotation files. This folder should be able to be opened and inspected. Approximate size 444 MB

6. Optional: VerifySubsetSize.py: Verifies that the size is correct in /sentencesCompressedSubset40K

7. TrainEvalTestSplit.py: Should split the /sentencesCompressedSubset40K into /trainEvalSet and /Testset. With approximate size /trainEvalset: 16.5 MB and /Testset: 428 MB.

8. AddPadding.py: Now the images in /sentencesCompressedSubset40k are padded to fit the final image size.

9. TxtFileGenerator.py: Should create Eval.txt, Train.txt and Test.txt in /txts !!Important to read the comments and change directories based on your drive path to "/Characterspotting".

10. ContaminationVerify.py: Make sure there are no common filenames between val, train and test.

11. Optional: Open "data_info.yaml" and change names, directories, nc(name count) etc as you see fit. Otherwise leave them as they are.


