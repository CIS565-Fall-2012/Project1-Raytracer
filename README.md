-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Sunday, 09/30/2012
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA capability! Any card after the Geforce 8xxx series will work. If you do not have an NVIDIA graphics card in the machine you are working on, feel free to use any machine in the SIG Lab or in Moore100 labs. All machines in the SIG Lab and Moore100 are equipped with CUDA capable NVIDIA graphics cards. If this too proves to be a problem, please contact Patrick or Karl as soon as possible.

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
In this project, you will implement a CUDA based raytracer capable of generating raytraced rendered images extremely quickly. For those of you who have taken CIS460/560, building a raytracer should not be anything new to you from a conceptual point of you. For those of you that have not taken CIS460/560, raytracing is a technique for generating images by tracing rays of light through pixels in an image plane out into a scene and following the way the rays of light bounce and interact with objects in the scene. More information can be found here: http://en.wikipedia.org/wiki/Ray_tracing_(graphics).

The ultimate purpose of this project is to serve as the foundation for your next project: a full CUDA based global illumination pathtracer. Raytracing can be thought of as a way to generate an isolated version of the direct light contribution in a global illumination scenario.

Since in this class we are concerned with working in generating actual images and less so with mundane tasks like file I/O, this project includes basecode for loading a scene description file format, described below, and various other things that generally make up the render "harness" that takes care of everything up to the rendering itself. The core renderer is left for you to implement. 
Finally, note that while this basecode is meant to serve as a strong starting point for a CUDA raytracer, you are not required to use this basecode if you wish, and you may also change any part of the basecode specification as you please, so long as the final rendered result is correct.

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project1 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio solution and the OSX makefile reference this folder for all source; the base source code compiles on OSX and Windows without modification.
* scenes/ contains an example scene description file.
* renders/ contains an example render of the given example scene file. 
* PROJ1_WIN/ contains a Windows Visual Studio 2010 project and all dependencies needed for building and running on Windows 7.
* PROJ1_OSX/ contains a OSX makefile, run script, and all dependencies needed for building and running on Mac OSX 10.8. 

The Windows and OSX versions of the project build and run exactly the same way as in Project0.

-------------------------------------------------------------------------------
REQUIREMENTS:
-------------------------------------------------------------------------------
In this project, you are given code for:

* Loading, reading, and storing the TAKUAscene scene description format
* Example functions that can run on both the CPU and GPU for generating random numbers, spherical intersection testing, and surface point sampling on cubes
* A class for handling image operations and saving images
* Working code for CUDA-GL interop

You will need to implement the following features:

* Raycasting from a camera into a scene through a pixel grid
* Phong lighting for one point light source
* Diffuse lambertian surfaces
* Raytraced shadows
* Cube intersection testing
* Sphere surface point sampling

You are also required to implement at least 2 of the following features:

* Specular reflection 
* Soft shadows and area lights 
* Texture mapping 
* Bump mapping
* Depth of field
* Supersampled antialiasing
* Refraction, i.e. glass
* OBJ Mesh loading and renderin
* Interactive camera

-------------------------------------------------------------------------------
BASE CODE TOUR:
-------------------------------------------------------------------------------
You will be working in three files: raytraceKernel.cu, intersections.h, and interactions.h. Within these files, areas that you need to complete are marked with a TODO comment. Areas that are useful to and serve as hints for optional features are marked with TODO (Optional). Functions that are useful for reference are marked with the comment LOOK.

* raytraceKernel.cu contains the core raytracing CUDA kernel. You will need to complete:
	* cudaRaytraceCore() handles kernel launches and memory management; this function already contains example code for launching kernels, transferring geometry and cameras from the host to the device, and transferring image buffers from the host to the device and back. You will have to complete this function to support passing materials and lights to CUDA.
	* raycastFromCameraKernel() is a function that you need to implement. This function once correctly implemented should handle camera raycasting. 
	* raytraceRay() is the core raytracing CUDA kernel; all of your raytracing logic should be implemented in this CUDA kernel. raytraceRay() should take in a camera, image buffer, geometry, materials, and lights, and should trace a ray through the scene and write the resultant color to a pixel in the image buffer.

* intersections.h contains functions for geometry intersection testing and point generation. You will need to complete:
	* boxIntersectionTest(), which takes in a box and a ray and performs an intersection test. This function should work in the same way as sphereIntersectionTest().
	* getRandomPointOnSphere(), which takes in a sphere and returns a random point on the surface of the sphere with an even probability distribution. This function should work in the same way as getRandomPointOnCube(). You can (although do not necessarily have to) use this to generate points on a sphere to use a point lights, or can use this for area lighting.

* interactions.h contains functions for ray-object interactions that define how rays behave upon hitting materials and objects. You will need to complete:
	* getRandomDirectionInSphere(), which generates a random direction in a sphere with a uniform probability. This function works in a fashion similar to that of calculateRandomDirectionInHemisphere(), which generates a random cosine-weighted direction in a hemisphere.
	* calculateBSDF(), which takes in an incoming ray, normal, material, and other information, and returns an outgoing ray. You can either implement this function for ray-surface interactions, or you can replace it with your own function(s).

You will also want to familiarize yourself with:

* sceneStructs.h, which contains definitions for how geometry, materials, lights, cameras, and animation frames are stored in the renderer. 
* utilities.h, which serves as a kitchen-sink of useful functions

-------------------------------------------------------------------------------
NOTES ON GLM:
-------------------------------------------------------------------------------
This project uses GLM, the GL Math library, for linear algebra. You need to know two important points on how GLM is used in this project:

* In this project, indices in GLM vectors (such as vec3, vec4), are accessed via swizzling. So, instead of v[0], v.x is used, and instead of v[1], v.y is used, and so on and so forth.
* GLM Matrix operations work fine on NVIDIA Fermi cards and later, but pre-Fermi cards do not play nice with GLM matrices. As such, in this project, GLM matrices are replaced with a custom matrix struct, called a cudaMat4, found in cudaMat4.h. A custom function for multiplying glm::vec4s and cudaMat4s is provided as multiplyMV() in intersections.h.

-------------------------------------------------------------------------------
TAKUAscene FORMAT:
-------------------------------------------------------------------------------
This project uses a custom scene description format, called TAKUAscene. TAKUAscene files are flat text files that describe all geometry, materials, lights, cameras, render settings, and animation frames inside of the scene. Items in the format are delimited by new lines, and comments can be added at the end of each line preceded with a double-slash.

Materials are defined in the following fashion:

* MATERIAL (material ID)								//material header
* RGB (float r) (float g) (float b)					//diffuse color
* SPECX (float specx)									//specular exponent
* SPECRGB (float r) (float g) (float b)				//specular color
* REFL (bool refl)									//reflectivity flag, 0 for no, 1 for yes
* REFR (bool refr)									//refractivity flag, 0 for no, 1 for yes
* REFRIOR (float ior)									//index of refraction for Fresnel effects
* SCATTER (float scatter)								//scatter flag, 0 for no, 1 for yes
* ABSCOEFF (float r) (float b) (float g)				//absorption coefficient for scattering
* RSCTCOEFF (float rsctcoeff)							//reduced scattering coefficient
* EMITTANCE (float emittance)							//the emittance of the material. Anything >0 makes the material a light source.

Cameras are defined in the following fashion:

* CAMERA 												//camera header
* RES (float x) (float y)								//resolution
* FOVY (float fovy)										//vertical field of view half-angle. the horizonal angle is calculated from this and the reslution
* ITERATIONS (float interations)							//how many iterations to refine the image, only relevant for supersampled antialiasing, depth of field, area lights, and other distributed raytracing applications
* FILE (string filename)									//file to output render to upon completion
* frame (frame number)									//start of a frame
* EYE (float x) (float y) (float z)						//camera's position in worldspace
* VIEW (float x) (float y) (float z)						//camera's view direction
* UP (float x) (float y) (float z)						//camera's up vector

Objects are defined in the following fashion:
* OBJECT (object ID)										//object header
* (cube OR sphere OR mesh)								//type of object, can be either "cube", "sphere", or "mesh". Note that cubes and spheres are unit sized and centered at the origin.
* material (material ID)									//material to assign this object
* frame (frame number)									//start of a frame
* TRANS (float transx) (float transy) (float transz)		//translation
* ROTAT (float rotationx) (float rotationy) (float rotationz)		//rotation
* SCALE (float scalex) (float scaley) (float scalez)		//scale

An example TAKUAscene file setting up two frames inside of a Cornell Box can be found in the scenes/ directory.

-------------------------------------------------------------------------------
BLOG
-------------------------------------------------------------------------------
As mentioned in class, all students should have student blogs detailing progress on projects. If you already have a blog, you can use it; otherwise, please create a blog using www.blogger.com or any other tool, such as www.wordpress.org. Blog posts on your project are due on the SAME DAY as the project, and should include:

* A brief description of the project and the specific features you implemented.
* A link to your github repo if the code is open source.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.  To create the video use http://www.microsoft.com/expression/products/Encoder4_Overview.aspx 

-------------------------------------------------------------------------------
THIRD PARTY CODE POLICY
-------------------------------------------------------------------------------
* Use of any third-party code must be approved by asking on Piazza.  If it is approved, all students are welcome to use it.  Generally, we approve use of third-party code that is not a core part of the project.  For example, for the ray tracer, we would approve using a third-party library for loading models, but would not approve copying and pasting a CUDA function for doing refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will result in you receiving an F for the semester.

-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Karl, yiningli@seas.upenn.edu, with a one paragraph explanation.  Be concise and realistic.  Recall that we reserve 30 points as a sanity check to adjust your grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We hope to only use this in extreme cases when your grade does not realistically reflect your work - it is either too high or too low.  In most cases, we plan to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as the path tracer.  We will determine the weighting at the end of the semester based on the size of each project.

-------------------------------------------------------------------------------
SUBMISSION
-------------------------------------------------------------------------------
As with the previous project, you should fork this project and work inside of your fork. Upon completion, commit your finished project back to your fork, and make a pull request to the master repository.
You should include a README.md file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running, and at least one screenshot of the final rendered output of your raytracer
* Instructions for building and running your project if they differ from the base code
* A link to your blog post detailing the project
* A list of all third-party code used