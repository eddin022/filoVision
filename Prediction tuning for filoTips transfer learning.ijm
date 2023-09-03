dir1 = getDirectory("Choose Source Directory ");
dir2 = getDirectory("Choose Prediction Directory");
dir3 = getDirectory("Choose Directory for Corrected Predictions");
list1 = getFileList(dir1);
list2 = getFileList(dir2);


setBatchMode(false);
for (i=0; i<list1.length; i++) {
 showProgress(i+1, list1.length);
 //print(list1[i]);
 open(dir1+list1[i]);
 sourceName = getTitle();
 run("Enhance Contrast", "saturated=0.35");
 setOption("ScaleConversions", true);
 run("8-bit");
 open(dir2+list2[i]);
 predictionName = getTitle();
 setMinAndMax(0, 2);
// run("Merge Channels...", "c1="+list1[i]+" c2="+list1[i]+);
 run("Merge Channels...", "c1='"+sourceName+"' c2='"+predictionName+"' create");
 waitForUser("Correct prediction pixel labels: ");   
 
 run("Split Channels");
 saveAs("TIFF", dir3+list1[i]);
 run("Close All");
}