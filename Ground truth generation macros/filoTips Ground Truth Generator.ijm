dir = getDirectory("Choose Source Directory: ");
tipPredictionDir = getDirectory("Choose Prediction Directory: ");
list = getFileList(dir);
erosionNum = 10
dilutionNum = 9
File.makeDirectory(dir+"/Masks/");

setBatchMode(false);
for (i=0; i<list.length; i++) {
 showProgress(i+1, list.length);
 //open image
 open(dir+list[i]);
 setOption("ScaleConversions", true);
 run("8-bit");
 run("Enhance Contrast", "saturated=0.35");
 run("Duplicate...", " ");
 bodyMask = getTitle();
 setAutoThreshold("Otsu dark no-reset");
 run("Threshold...");
 waitForUser;
 run("Convert to Mask");
 run("Fill Holes");
 for (j=0; j<erosionNum; j++) {
  run("Erode");
 };
 for (j=0; j<dilutionNum; j++) {
  run("Dilate");
 };
 run("Divide...", "value=255");
 setMinAndMax(0, 2);
 run("Merge Channels...", "c2=["+list[i]+"] c4=["+bodyMask+"] create keep");
 run("Arrange Channels...", "new=21");
 run("Previous Slice [<]");
 setTool("Paintbrush Tool");
 run("Paintbrush Tool Options...", "brush=40");
 waitForUser("Make any final edits to the body mask!: ");
 run("Split Channels");
 selectImage("C1-Composite");
 bodyMask = "C1-Composite";
 
 //read in tip predictions
 open(tipPredictionDir+list[i]);
 tipMask = getImageID();
 run("Subtract...", "value=1");
 run("Multiply...", "value=2");
 imageCalculator("Add create",tipMask, bodyMask);
 final = getTitle();
 run("Max...", "value=2");
 setMinAndMax(0, 2);
 close("C1-Composite");
 close("C2-Composite");
 run("Merge Channels...", "c2=["+list[i]+"] c4=["+final+"] create keep");
 run("Arrange Channels...", "new=21");
 run("Previous Slice [<]");
 setTool("Paintbrush Tool");
 run("Paintbrush Tool Options...", "brush=40");
 waitForUser("Make any final edits!: ");
 run("Split Channels");
 final = "C1-Composite";
 selectImage(final);
 run("Select None");
 saveAs("TIFF", dir+"/Masks/"+list[i]);
 run("Close All");
}