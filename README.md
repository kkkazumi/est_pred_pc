#mental estimation prog tutorial

####$B7'C+=$O@<B83(B
####$B%/%$%:%2!<%`$GF@$i$l$?%G!<%?$+$i5$J,?dDj$r9T$&%W%m%0%i%`(B

##$BFbMF!'(B
####README.md
* $B@bL@$,=q$$$F$"$j$^$9!#(B

####factor.csv
* $B0x;R72$N%G!<%?(B

	+ $B%/%$%:%2!<%`$N>u67$H%m%\%C%H9TF0$rI=$9(B

	+ $B3FJQ?t(B
		+ $B%/%$%:2s?t!"@5EzN(!"6&469TF0<B;\N(!"Ne$^$79TF0<B;\N(!"$+$i$+$$9TF0<B;\N(!"%/%$%:$H4X78$J$$9TF0<B;\N(!"H/OC!&F0:nL5$73NN(!"3MF@E@?t!"O">!?t!"O"GT?t(B

####signal.csv

* $BI=>pG'<17k2L$N%G!<%?72(B

	- $B4n$S!"6C$-!"E\$j!"Ha$7$_$NI=>p$,A4BN$NI=>p$K@j$a$k3d9g$rI=$9(B
	- $B3FJQ?t(B
		- $B4n$S!"6C$-!"E\$j!"Ha$7$_(B

####estimate.cpp

* `factor.csv`$B$H(B`signal.csv`$B$+$i5$J,?dDj7k2L$N;~7ONs%G!<%?$r;;=P$9$k%W%m%0%i%`(B
* $B;;=P$7$?5$J,?dDj7k2L$N;~7ONs%G!<%?$O(B`mental.csv` $B$K=PNO$5$l$k(B

####predict.cpp
* `factor.csv`$B$H(B`mental.csv`$B$+$i5$J,JQ2=M=B,7k2L$N;~7ONs%G!<%?$r;;=P$9$k%W%m%0%i%`(B
* $B;;=P$7$?5$J,JQ2=M=B,7k2L$N;~7ONs%G!<%?$O$"$k%U%!%$%k(B($BL>A0$O$^$@L5$$(B)$B$K=PNO$5$l$k(B

##$B%W%m%0%i%`$rF0$+$9=`Hw(B
* Eigen$B$r%$%s%9%H!<%k(B

	```	
	sudo apt-get install libeigen3-dev #$B%P!<%8%g%s$OG$0U(B
	```

*  gcc$B$N:G?7HG$r%$%s%9%H!<%k$7$F$*$/(B

	```
	$sudo apt-get install g++-4.8
	```

##$B%W%m%0%i%`$N$&$4$+$7J}(B

1. `estimate.cpp`$B$N%3%s%Q%$%k(B

	```
	$g++ estimate.cpp -std=c++11 -o estimate
	```

2. `estimate` $B$r<B9T$9$k(B
	- $B5$J,?dDj7k2L$r=PNO$9$k%U%!%$%kL>$r0z?t$K$9$k(B

	```
	$./estimate test
	```
	
	- `/test

##$B5$J,?dDj7k2L$HHo83<T<+?H$K$h$k5$J,I>2A7k2L$H$NAj4X7W;;(B

###sokan.R

1. `R` $B$rN)$A>e$2$k(B

2. `sokan.R` $B$rFI$_9~$`!#(B `source('sokan.R')`

3. `q()` $B$G(B `R` $B$+$i%7%'%k$KLa$k!#(B

4. `soukan.csv` $BFb$K=PNO$5$l$F$$$k$h!#(B#($B2r@bI,MW$J$/$M!&!&!)(B)


## func_new.py

- contains new functions to calculate again.