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


##Contacting the author
ASOP was written by Boris Gorelik. Author's personal site is
[http://gorelik.net][7], author's e-mail is [boris@gorelik.net][4]



[1]: http://onlinelibrary.wiley.com/doi/10.1002/prot.21847/full
[2]: http://en.wikipedia.org/wiki/Particle_filter
[3]: http://en.wikipedia.org/wiki/Genetic_algorithm
[4]: mailto:boris@gorelik.net
[5]: https://github.com/bgbg/asop/tags
[6]: https://github.com/bgbg/randomArbitrary
[7]: http://gorelik.net



