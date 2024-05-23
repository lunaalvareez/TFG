

Training:

Create main folder "Characterspotting" on your computer.

Create the subfolders "/sentencesCompressedOriginal", /sentencesCompressed, "/sentencesCompressedSubset40K", "/txts", "/trainEvalSet", "/testSet"

Get "words", "xml", "trainset.txt" from IAMHandwriting dataset. Place them in your root directory (/Characterspotting)

Before each run. Make sure to read the comments and change the paths as necessary. Run them in accordance with the order.
Run:
1. Compressor.py: and make sure /sentencesCompressedOriginal is filled with around 1.6GB. If you can open the folder and look through that's good but it's not needed if it's not possible.

2. AddPadding.py: Now the images in /sentencesCompressed are padded to fit the final image size.

3. AnnotationGenerator.py <--padded>: Now the /sentencesCompressed should have annotation files accompanying each image. If the images have been padded, the flag --padded should be added.

4. BalanceData.py: If the data has been padded, the folder of origin would be sentencesCompressed and if the data is not padded, sentencesCompressedOriginal.
                    The output folder will always be sentencesCompressed.

5. Optional: RemoveSmallImages.py: Removes all images and annotation files with width or height smaller than 10 in /sentencesCompressed. Also removes images with errors being opened. Decreases /wordscompressed by 5%.

6. Optional: RemoveSingleDigits.py: Removes all Images and annotation files with single digits in /sentencesCompressed. Decreases /sentencesCompressed by 11%.

7. TrainEvalTestSplit.py: Should split the /sentencesCompressedSubset40K into /trainEvalSet and /testSet. With approximate size /trainEvalset: 16.5 MB and /Testset: 428 MB.

8. TxtFileGenerator.py: Should create Eval.txt, Train.txt and Test.txt in /txts !!Important to read the comments and change directories based on your drive path to "/Characterspotting".

9. ContaminationVerify.py: Make sure there are no common filenames between val, train and test.

10. Optional: Open "data_info.yaml" and change names, directories, nc(name count) etc as you see fit. Otherwise leave them as they are.


