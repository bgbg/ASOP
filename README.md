# ASOP (Adaptive Search for Optimal Populations) optimization algorithm

## What is ASOP?
This is a Python implementation of ASOP -- Adaptive Search for Optimal
Populations. ASOP is a general global optimization algorithm, that was
partially inspired by [ISE][1], [Particle filter][2], [Genetic Algorithms][3]
and other techniques and algorithms.

This is a work in progress. Unfortunately, almost no documentation is
available yet. You are more than welcome to [contact the author][4] for more
details, help and help suggestions.

## Cloning ASOP
The best way to obtain this library is to go its [Github's tags page][5] and
to download the latest tag.

If you decided to clone ASOP, bare in mind that ASOP repository contains
a submodule ([randomArbitrary][6] -- lirandom number generators for arbitrary
distributions). Therefore, the proper way of cloning ASOP is

	git clone --recursive <path-to-ASOP>

If you forgot to clone recursively, you will see an empty directory
named `randomArbitrary`. Don't worry. Just go to the ASOP
directory, and run the following command:

	git submodule update --init


##Contacting the author, consulting
ASOP was written by Boris Gorelik. Author's personal site is
[http://gorelik.net][7], author's e-mail is [boris@gorelik.net][4]. 
As written above, you are
more than welcome to contact the author with any question, suggestion
or thoughts. 

## License

This software is distributed under the MIT license (see full license text 
below). You are not required to, but it will be very nice of you if you 
acknowledge the author, Boris Gorelik and even the link to his personal
site ([http://gorelik.net][7]). 

Copyright (C) 2012 Boris Gorelik

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

[1]: http://onlinelibrary.wiley.com/doi/10.1002/prot.21847/full
[2]: http://en.wikipedia.org/wiki/Particle_filter
[3]: http://en.wikipedia.org/wiki/Genetic_algorithm
[4]: mailto:boris@gorelik.net
[5]: https://github.com/bgbg/asop/tags
[6]: https://github.com/bgbg/randomArbitrary
[7]: http://gorelik.net



