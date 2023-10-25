dir1 = getDirectory("Choose Input Directory ");
dir2 = getDirectory("Choose Output Directory");
list1 = getFileList(dir1);

setBatchMode(false);
for (i=0; i<list1.length; i++) {
 showProgress(i+1, list1.length);
 //print(list1[i]);
 open(dir1+list1[i]);
 sourceName = getTitle();
 print(sourceName);
 run("Enhance Contrast", "saturated=0.35");
 run("Duplicate...", " ");
 filoStalkDup = getTitle();
 print(filoStalkDup);
 run("Make Binary");
 run("Fill Holes");
 waitForUser("Fill in cell body edges as needed: ");  
 run("Fill Holes");
 waitForUser("Last chance to correct cell body edges: ");  
 run("Fill Holes");
 run("Duplicate...", " ");
 bodyDup = getTitle();
 print(bodyDup);
 run("Erode");
 run("Erode");
 run("Erode");
 run("Erode");
 run("Erode");
 run("Erode");
 run("Dilate");
 run("Dilate");
 run("Dilate");
 run("Dilate");
 run("Dilate");
 run("Dilate");
 run("Max...", "value=1");
 selectWindow(filoStalkDup);
 run("Max...", "value=2");
 imageCalculator("Subtract create", filoStalkDup, bodyDup);
 saveAs("TIFF", dir2+list1[i]);
 run("Close All");
}




