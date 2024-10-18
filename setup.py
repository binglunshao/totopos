import setuptools

setuptools.setup(
	name = "totopos", 
	version = "0.0.1", 
	author = "Emanuel Flores", 
	author_email = "manuflores {at} caltech {dot} edu",
	description = "Trajectory outlining for topological loops in single-cell data", 
	packages = setuptools.find_packages(), 
	classifiers = [
		"Programming Language :: Python :: 3", 
		"License :: OSI Approved :: MIT License", 
		"Operating System :: OS Independent"
	]
)