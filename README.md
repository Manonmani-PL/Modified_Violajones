# Modified_Violajones
First Cascade classifier is used like Viola Jones algorithm 

<p align="center">
  <img src="https://github.com/Manonmani-PL/Modified_Violajones/blob/master/data/Untitled%20document-1%20(1).jpg" width="350" title="hover text">
  
</p>


<li> As a next step filters are added to adaboost as most of the manga drawings will be in line(sketch based), viola jones method is not that much effective for this kind of pages. </li>
<li> Integral image is taken before we add three filters to it first is Linear edge detector, Next is average intensity detector and the third is local edge orientation histogram(EOH) </li>
<li> And then sobel mask is added to it </li> 
<li> Finally cascade of classifier is given to SVM for classification purpose </li>
<p>The final accuracy for this model is 81% which is better than viola jones algorithm. perks of using this algorithm is we don’t want that much big data sets, to make the model learn very few is more than enough </p>
