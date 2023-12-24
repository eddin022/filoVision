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
 run("Enhance Contrast", "saturated=0.99");
 setOption("ScaleConversions", true);
 run("8-bit");
 run("Duplicate...", " ");
 filoStalkDup = getTitle();
 print(filoStalkDup);
 run("Convert to Mask");
 run("Fill Holes");
 run("Merge Channels...", "c2="+sourceName+" c4="+filoStalkDup+" create ignore");
 setMinAndMax(0, 50);
 waitForUser("Outline cell body edges as needed with paintbrush tool. Will fill holes automatically. Remember to select correct channel and color: ");  
 run("Split Channels");
 selectWindow("C2-Composite");
 run("Convert to Mask");
 run("Fill Holes");
 run("Merge Channels...", "c2=C1-Composite c4=C2-Composite create ignore");
 waitForUser("Last chance to correct cell body edges: ");  
 run("Split Channels");
 selectWindow("C2-Composite");
 run("Convert to Mask");
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
 run("Merge Channels...", "c2=C1-Composite c4=C2-Composite create ignore");
 selectWindow("Composite");
 waitForUser("Correct filopodia stalks as needed: ");
 run("Split Channels");
 selectWindow("C2-Composite");
 run("Max...", "value=2");
 imageCalculator("Subtract create", "C2-Composite", "C2-Composite-1");
 saveAs("TIFF", dir2+list1[i]);
 run("Close All");
}




