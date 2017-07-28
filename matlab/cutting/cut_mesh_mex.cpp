/*
	[Vc,Fc] = cut_mesh_mex(V,F);
    V, F: vertices/faces of input mesh (any topology, with/without boundary)
    Vc, Fc: vertices/faces of cut mesh
*/

#include <iostream>
#include <stdlib.h>     /* srand, rand */

#include "mex.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <igl/matlab_format.h>
#include <igl/adjacency_matrix.h>
#include <igl/boundary_loop.h>
#include "polyvector_field_cut_mesh_with_singularities_randomized.h"
#include <igl/cut_mesh.h>
#include <igl/euler_characteristic.h>

using namespace std;

void mexFunction(	int nlhs, mxArray *plhs[], 
				 int nrhs, const mxArray*prhs[] ) 
{ 
	/* retrieve arguments */
	if( nrhs!=2 ) 
		mexErrMsgTxt("2 input arguments are required - faces and vertices."); 
	if( nlhs<2 ) 
		mexErrMsgTxt("2 output arguments are required - faces and vertices."); 

	// first argument : vertices
    double *V_mex = mxGetPr(prhs[0]);
    int nV = (int)mxGetM(prhs[0]);
    int cV = (int)mxGetN(prhs[0]);    
	if( cV!=3 ) 
		mexErrMsgTxt("Vertices should be an #V x 3 matrix.");     
    Eigen::MatrixXd V = Eigen::Map<Eigen::MatrixXd>(V_mex,nV,3);

	// second argument : faces
    double *F_mex = mxGetPr(prhs[1]);
    int nF = (int)mxGetM(prhs[1]);
    int cF = (int)mxGetN(prhs[1]);    
	if( cF!=3 ) 
		mexErrMsgTxt("Faces should be an #F x 3 matrix.");     
    Eigen::MatrixXd Fd = Eigen::Map<Eigen::MatrixXd>(F_mex,nF,3);
    Eigen::MatrixXi F = Fd.cast<int> ();
    // C-based indexing
    F = F.array() -1;
      
    // compute Euler characteristic of input mesh
    int e_in = igl::euler_characteristic(V, F);
    // compute number of boundaries in input mesh
    std::vector<std::vector<int> > L;
    igl::boundary_loop(F,L);    
    int b_in = L.size();
    // compute genus of input mesh
    int g_in =(2-b_in-e_in)/2;
    cerr<<" -> Input mesh: e = "<<e_in<<", b = "<<b_in<<", g= "<<g_in<<endl;
    
    // for genus zero we need to add two non-adjacent singularities
    Eigen::VectorXi singularities;
    if (g_in ==0)
    {
        // generate two random singularities between 0 and nV-1:
        int s0 = rand() % nV ;
        // make sure singus aren't adjacent
        Eigen::SparseMatrix<int> A;
        igl::adjacency_matrix(F,A);        
        int s1 = s0;        
        while (s1==s0 || A.coeff(s0,s1) == 1)
            s1 = rand() % nV ;
        
        singularities.setZero(2,1);
        singularities<<s0,s1;
    }
    
        
        
    // generate cuts using tree traversal: a boolean per face edge
    Eigen::MatrixXi cuts;
    polyvector_field_cut_mesh_with_singularities_randomized(V, F, singularities, cuts);
    
    // duplicate vertices along cut to produce cut mesh
    Eigen::MatrixXd Vc;
    Eigen::MatrixXi Fc;
    igl::cut_mesh(V, F, cuts, Vc, Fc);

    // compute Euler characteristic of output mesh
    // this should be 1 ALWAYS (disk topology)
    int e_out = igl::euler_characteristic(Vc, Fc);
    // compute number of boundaries in input mesh
    std::vector<std::vector<int> > Lc;
    igl::boundary_loop(Fc,Lc);    
    int b_out = Lc.size();
    // compute genus of input mesh
    int g_out =(2-b_out-e_out)/2;
    cerr<<" -> Output mesh: e = "<<e_out<<", b = "<<b_out<<", g= "<<g_out<<endl;

    if( e_out!=1 ) 
		mexErrMsgTxt("Output mesh does not have disk topology."); 

	// first output : vertices of cut mesh
	plhs[0] = mxCreateDoubleMatrix(Vc.rows(), 3, mxREAL); 
    Eigen::Map<Eigen::MatrixXd>( mxGetPr(plhs[0]), Vc.rows(), Vc.cols() ) = Vc;
    
	// second output : faces of cut mesh
    Eigen::MatrixXd Fc_d = Fc.cast<double>();
    // matlab-based indexing
    Fc_d = Fc_d.array() + 1;    
	plhs[1] = mxCreateDoubleMatrix(Fc.rows(), 3, mxREAL); 
    Eigen::Map<Eigen::MatrixXd>( mxGetPr(plhs[1]), Fc_d.rows(), Fc_d.cols() ) = Fc_d;

	return;
}
