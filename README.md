# Geometric Non-interference Constraint Formulation
## INTRO
The geometric non-interference constraint can be represented by a level set function that approximates the signed distance function (SDF). Our method generates a Bspline that locally approximates the SDF by minimized 3 energy terms. The energy minimization is performed in Python by using the OpenMDAO optimization framework, coupled with the use of PyOptSparse, a framework for formulating optimization problems, SNOPT, a gradient-based optimization algorithm, and lsdo_geo, a cython representation of Bsplines.

## GETTING STARTED
To begin, you must have a point cloud representation of your boundary (2D or 3D) with unit normal vectors pointing the direction of infeasible space. To perform optimization, you must have OpenMDAO, PyOptSparse, SNOPT, and lsdo_geo installed. SNOPT is requires a license obtain.

## RUNNING THE OPTIMIZATION
With your point cloud ready, navigate to the 2D or 3D folders depending on your structure. Follow the steps in the run.py file, where the configurations are edittable at the top of the file. Run this file and save the "Func" object, which contains the functions to evaluate the Bspline, coordinate system, minimum bounding box information, and other data to quantify the results.

## OPTIONAL
If you are familiar with CSDL, there is an all-in-one CSDL model that runs the optimization under the 'csdl_implementation' folder.

## CITATION
Paper is submitted and under review.

## FIGURES IN THE PAPER
Figures generated for the paper are located under the 'Generate_Figures' folder, with the necessary pickle data and run scripts to generate them.